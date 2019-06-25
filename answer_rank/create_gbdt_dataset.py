import os
import json
import configargparse
import operator
from tqdm import tqdm
from answer_rank.eval import calc_f1, calc_bleu
import random
import pickle
import pandas as pd
# random.seed(2019)
import os
import answer_rank.config

def to_pandas(data_dir, df):
    file_names = os.listdir(data_dir)
    for path in file_names:
        if not path.endswith("txt"):
            continue
        preds = []
        with open(os.path.join(data_dir,path), "r") as fr:
            for line in fr:
                lines = line.strip("\n")
                preds.append(lines)
        df[path] = preds
    print(df.loc[0])
    return df


def make_samples(predict_dir, target_file, topic_file, src_file, sample_save, is_training):
    df = pd.DataFrame()
    df = to_pandas(predict_dir, df)

    columns = df.columns.tolist()

    for c in columns:
        df[c] = df[c].apply(lambda s: s + "\t" + str(config.model_weight_dict[c] / 9054))

    print("shape of df: ", df.shape)

    fw = open(sample_save, 'w')

    size = df.shape[0]

    topics, target, source = [], [], []

    with open(topic_file, 'r') as fr:
        for topic in fr:
            topics.append(eval(topic.strip("\n")))

    if is_training:
        with open(target_file, 'r') as fr:
            for tgt in fr:
                target.append(tgt)
    else:
        target = ["<blank>"] * len(topics)
        print("len of target", len(topics))

    with open(src_file, 'r') as fr:
        for src in fr:
            source.append(src)

    for i, src, tgt, topic in zip(range(size), source, target, topics):
        src = src.strip("\n")

        tgt = tgt.strip("\n")
        tgt = tgt.replace("<SOS> ", "")
        tgt = tgt.replace(" <EOS>", "")

        preds = []
        data = df.loc[i].tolist()
        for ix, d in enumerate(data):
            d = d.replace("<SOS> ", "").replace(" <EOS>", "")
            for k, v in topic.items():
                d = d.replace(k, v)
            preds.append(d)

        topic_list = sorted(topic.items(), key=lambda item: len(item[1]), reverse=True)
        for k, v in topic_list:
            tgt = tgt.replace(k, v)
            src = src.replace(k, v)

        sample = {"preds": preds, "tgt": tgt, "src": src}
        print(json.dumps(sample), file=fw)
    fw.close()



def create_rank_dataset(sample_save, dataset_save, is_training):
    fw = open(dataset_save, 'w')
    fw.write("preds\tbeam_score\tmodel_weight\ttgt\tsrc\tscore\n")
    best_fw = open("best_dev.txt", 'w')
    with open(sample_save, 'r') as fr:
        for line in tqdm(fr, desc="CreateDataset"):
            sample = json.loads(line.strip("\n"))
            preds = sample["preds"]
            tgt = sample["tgt"]

            if is_training:
                result = []
                for ix, pred in enumerate(preds):
                    pred = pred.split("\t")[0]
                    f1 = calc_f1([[pred, tgt]])
                    bleu1, bleu2 = calc_bleu([[pred, tgt]])
                    score = f1 + bleu1 + bleu2
                    result.append((score, ix))
                sorted_res = sorted(result, key=operator.itemgetter(0), reverse=True)
                best_fw.write(preds[sorted_res[0][1]]+"\n")
                for res in result:
                    fw.write(preds[res[1]] + "\t" + tgt + "\t" + sample["src"] + "\t" + str(res[0]) + "\n")
            else:
                for ix, pred in enumerate(preds):
                    fw.write(pred + "\t" + tgt + "\t" + sample["src"] + "\t" + '-1' + "\n")
    fw.close()
    best_fw.close()


def dataset_opt(parser):
    parser.add_argument("--predict_dir", "-predict_dir", type=str, default="/home/daizelin/kbqa-onmt/outputs/result/test_2/")
    parser.add_argument("--target_file", "-target_file", type=str, default="data/dev.tgt")
    parser.add_argument("--topic_file", "-topic_file", type=str, default="data/test_topic.txt")
    parser.add_argument("--src_file", "-src_file", type=str, default="data/test.src")
    parser.add_argument("--sample_save", "-sample_save", type=str, default="data/samples.json")
    parser.add_argument("--dataset_save", "-dataset_save", type=str, default="/home/daizelin/answer_rank/data/test_raw.txt")
    parser.add_argument("--is_training", "-is_training", type=bool, default=False)


def main(opt):
    make_samples(opt.predict_dir, opt.target_file, opt.topic_file, opt.src_file, opt.sample_save, opt.is_training)
    create_rank_dataset(opt.sample_save, opt.dataset_save, opt.is_training)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    dataset_opt(parser)
    opt = parser.parse_args()
    main(opt)