import os
import json
import configargparse
import operator
from tqdm import tqdm
from eval import calc_f1, calc_bleu
import random
# random.seed(2019)


def make_samples(predict_dir, target_file, topic_file, src_file, sample_save):
    fw = open(sample_save, 'w')
    topics, target, source = [], [], []
    file_names = os.listdir(predict_dir)
    assert len(file_names) == 4

    with open(topic_file, 'r') as fr:
        for topic in fr:
            topics.append(eval(topic.strip("\n")))

    with open(target_file, 'r') as fr:
        for tgt in fr:
            target.append(tgt)

    with open(src_file, 'r') as fr:
        for src in fr:
            source.append(src)

    pred0_path = os.path.join(predict_dir, file_names[0])
    pred1_path = os.path.join(predict_dir, file_names[1])
    pred2_path = os.path.join(predict_dir, file_names[2])
    pred3_path = os.path.join(predict_dir, file_names[3])

    pred0_fr =  open(pred0_path, 'r')
    pred1_fr = open(pred1_path, 'r')
    pred2_fr = open(pred2_path, 'r')
    pred3_fr = open(pred3_path, 'r')

    data = zip(pred0_fr, pred1_fr, pred2_fr, pred3_fr, source, target, topics)

    for pred0, pred1, pred2, pred3, src, tgt, topic in data:
        src = src.strip("\n")

        tgt = tgt.strip("\n")
        tgt = tgt.replace("<SOS> ", "")
        tgt = tgt.replace(" <EOS>", "")

        pred0 = pred0.strip("\n")
        pred0 = pred0.replace("<SOS> ", "")
        pred0 = pred0.replace(" <EOS>", "")

        pred1 = pred1.strip("\n")
        pred1 = pred1.replace("<SOS> ", "")
        pred1 = pred1.replace(" <EOS>", "")

        pred2 = pred2.strip("\n")
        pred2 = pred2.replace("<SOS> ", "")
        pred2 = pred2.replace(" <EOS>", "")

        pred3 = pred3.strip("\n")
        pred3 = pred3.replace("<SOS> ", "")
        pred3 = pred3.replace(" <EOS>", "")

        for k, v in topic.items():
            tgt = tgt.replace(k, v)
            src = src.replace(k, v)
            pred0 = pred0.replace(k, v)
            pred1 = pred1.replace(k, v)
            pred2 = pred2.replace(k, v)
            pred3 = pred3.replace(k, v)

        preds = [pred0, pred1, pred2, pred3]
        sample = {"preds": preds, "tgt": tgt, "src": src}
        print(json.dumps(sample), file=fw)

    pred0_fr.close()
    pred1_fr.close()
    pred1_fr.close()
    pred3_fr.close()
    fw.close()


def train_val_split(X, y, valid_size=0.2, random_state=2018, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    # logger.info('train val split')

    train, valid = [], []
    bucket = [[] for _ in range(2)]

    for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
        bucket[int(data_y)].append((data_x, data_y))

    del X, y

    for bt in tqdm(bucket, desc='split'):
        N = len(bt)
        if N == 0:
            continue
        test_size = int(N * valid_size)

        if shuffle:
            random.seed(random_state)
            random.shuffle(bt)

        valid.extend(bt[:test_size])
        train.extend(bt[test_size:])

    if shuffle:
        random.seed(random_state)
        random.shuffle(valid)
        random.shuffle(train)

    return train, valid


def create_rank_dataset(sample_save, dataset_save):
    fw = open(dataset_save, 'w')
    # fw.write("preds\ttgt\tsrc\trank\n")
    examples = []
    labels = []
    with open(sample_save, 'r') as fr:
        for line in tqdm(fr, desc="CreateDataset"):
            sample = json.loads(line.strip("\n"))
            preds = sample["preds"]
            tgt = sample["tgt"]

            result = []
            for ix, pred in enumerate(preds):
                f1 = calc_f1([[pred, tgt]])
                bleu1, bleu2 = calc_bleu([[pred, tgt]])
                score = f1 + bleu1 + bleu2
                result.append((score, ix))
            sorted_res = sorted(result, key=operator.itemgetter(0), reverse=True)
            # ranks = [str(w[1]) for w in sorted_res]
            # fw.write("|".join(preds) + "\t" + tgt + "\t" + sample["src"] + "\t" + "|".join(ranks) + "\n")
            for index, res in enumerate(sorted_res):
                if index == 0:
                    label = 1
                else:
                    label = 0

                example = [{"text_b": preds[res[1]], "text_a": sample["src"].replace("[CLS]", "[START]"), "label":label}]

                labels.append(label)
                examples.append(example)

            example_2 = [{"text_b": tgt, "text_a": sample["src"].replace("[CLS]", "[START]"), "label": 1}]
            label_2 = 1
            labels.append(label_2)
            examples.append(example_2)

    train, valid = train_val_split(examples, labels)
    fw_1 = open("/home/daizelin/QA_match/data/dev.json", 'w')
    for data in train:
        example = data[0]
        print(json.dumps(example), file=fw)

    for data in valid:
        example = data[0]
        print(json.dumps(example), file=fw_1)
    fw.close()
    fw_1.close()


def dataset_opt(parser):
    parser.add_argument("--predict_dir", "-predict_dir", type=str, default="data/predicts/")
    parser.add_argument("--target_file", "-target_file", type=str, default="data/dev.tgt")
    parser.add_argument("--topic_file", "-topic_file", type=str, default="data/dev_topic.txt")
    parser.add_argument("--src_file", "-src_file", type=str, default="data/dev.src")
    parser.add_argument("--sample_save", "-sample_save", type=str, default="data/samples.json")
    parser.add_argument("--dataset_save", "-dataset_save", type=str, default="/home/daizelin/QA_match/data/train.json")



if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    dataset_opt(parser)
    opt = parser.parse_args()
    make_samples(opt.predict_dir, opt.target_file, opt.topic_file, opt.src_file, opt.sample_save)
    create_rank_dataset(opt.sample_save, opt.dataset_save)