import os
import json
from tqdm import tqdm
import configargparse
import collections
from onmt.utils.logging import init_logger, logger
#from pyltp import Postagger, NamedEntityRecognizer

CONFIG_FILE = "./config/preclean.yml"


#postagger = Postagger()
#postagger.load("ltp_model/pos.model")


# def ltp_pos(inputs):
#     inputs = inputs.split()
#     pos_inputs = []
#     postags = list(postagger.postag(inputs))
#     for d, p in zip(inputs, postags):
#         if d.startswith("[Q=") or d.startswith("[A=") or d.startswith("[CLS]") or d.startswith("[Goal=") or d=="[KG]":
#             p = "special"
#         pos_inputs.append(d + "|" + p)
#     # print(pos_inputs)
#     return " ".join(pos_inputs)


invalid_kg = []
with open("data/invalid_kg.json", 'r') as fr:
    for line in fr:
        kg = json.loads(line.strip("\n"))
        invalid_kg.append(kg)
print("invalid_kg list length:%d"%len(invalid_kg))


def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    fout = open(sample_file, 'w')
    with open(session_file, 'r') as f:
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)
            conversation = session["conversation"]

            for j in range(0, len(conversation), 2):
                sample = collections.OrderedDict()
                sample["goal"] = session["goal"]
                sample["knowledge"] = session["knowledge"]
                sample["history"] = conversation[:j]
                sample["response"] = conversation[j]
                sample["invalid_kg"] = invalid_kg[i]
                # print(sample)
                sample = json.dumps(sample, ensure_ascii=False)
                fout.write(sample + "\n")

    fout.close()


def test_input_select(history, knowledge, goal):
    def _compute_overlap_num(tgt, kg):
        if isinstance(tgt, str):
            tgt = list(set(tgt.split()))
        if isinstance(kg, str):
            kg = list(set(kg.split()))
        overlap_num = 0
        for t in tgt:
            for k in kg:
                if t == k:
                    overlap_num += 1
        return overlap_num

    def get_words(kg):
        words = []
        for item in kg:
            for w in item.split():
                words.append(w)
        return words

    new_knowledge = []
    new_goal = []
    history = "".join(history)
    if history == "":
        new_goal.append(goal[0])
        for kg in knowledge:
            kg = get_words(kg)
            _goal = get_words(goal[0])
            overlap_num = _compute_overlap_num(_goal, kg)
            if overlap_num > 0:
                new_knowledge.append(kg)
    else:
        # 使用goal作为kg的选取参考
        new_goal = goal
        # 首先统计目标goal和在历史对话中是否出现
        # 选出出现过的goal，用他来选取kg
        _goal = []

        for g in goal:
            g = get_words(g)
            for h in history:
                overlap_num = _compute_overlap_num(h, g)
                if overlap_num > 0:
                    _goal.append(g)
                    break

        if len(_goal) > 0:
            for kg in knowledge:
                kg = get_words(kg)
                for g in _goal:
                    g = get_words(g)
                    overlap_num = _compute_overlap_num(g, kg)
                    if overlap_num > 0:
                        new_knowledge.append(kg)
                        break
        else:
            # 如果goal都没出现过的话，那么拿第一个goal去选取kg
            new_goal = [goal[0]]
            g = get_words(goal[0])

            for kg in knowledge:
                kg = get_words(kg)
                overlap_num = _compute_overlap_num(g, kg)
                if overlap_num > 0:
                    new_knowledge.append(kg)
    # print(new_knowledge)
    # print("*", new_goal)
    return new_knowledge, new_goal


def preprocessing_for_one_conversation(text,
                                       topic_generalization,
                                       augmentation,
                                       corpus_type):
    """
    preprocessing_for_one_conversation
    """
    conversation = json.loads(text.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)

    goal = conversation["goal"]
    knowledge = conversation["knowledge"]
    history = conversation["history"]
    response = conversation["response"] if "response" in conversation else "null"

    topic_a = goal[0][1]
    topic_b = goal[0][2]
    for i, [s, p, o] in enumerate(knowledge):
        if u"领域" == p:
            if topic_a == s:
                domain_a = o
            elif topic_b == s:
                domain_b = o

    topic_dict = {}
    if u"电影" == domain_a:
        topic_dict["video_topic_a"] = topic_a
    else:
        topic_dict["person_topic_a"] = topic_a

    if u"电影" == domain_b:
        topic_dict["video_topic_b"] = topic_b
    else:
        topic_dict["person_topic_b"] = topic_b

    # if corpus_type != "train":
    #     knowledge, goal = test_input_select(history, knowledge, goal)
    chat_path_str = ' '.join([' '.join(["[Goal=%d]"%ix] + spo) for ix, spo in enumerate(goal)])
    knowledge_str1 = ' '.join([' '.join(["[KG]"] + spo) for spo in knowledge])

    if augmentation:
        invalid_kg = conversation["invalid_kg"]
        knowledge_str2 = ' '.join([' '.join(["[KG]"] + spo) for spo in knowledge if spo not in invalid_kg])
    # knowledge_str2 = '\1'.join([' '.join(spo) for spo in knowledge])
    # history_str = ' '.join(history)
    processed_history = ["[Q=0] [CLS]"]

    turns = 1
    for ix, h in enumerate(history):
        if ix %2 ==0:
            if ix == 0:
                processed_history.append("[A=0]")
            else:
                processed_history.append("[A=%d]"%turns)
                turns +=1
        else:
            processed_history.append("[Q=%d]"%(turns))
        processed_history.append(h)

    history_str = " ".join(processed_history)

    src = chat_path_str + " " + knowledge_str1 + " " + history_str
    # model_text = '\t'.join([src, response, knowledge_str2])
    tgt = "<SOS> " + response + " <EOS>"
    source, target = [], []
    # source.append(src)
    # target.append(tgt)

    if augmentation:
        src_2 = chat_path_str + " " + knowledge_str2 + " " + history_str
        source.append(src_2)
        target.append(tgt)

    if topic_generalization:
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for key, value in topic_list:
            src = src.replace(value, key)
            tgt = tgt.replace(value, key)
            if augmentation:
                src_2 = src_2.replace(value, key)

        if augmentation:
            source.append(src_2)
            target.append(tgt)

        source.append(src)
        target.append(tgt)
    return source, target, topic_dict


def convert_conversation_corpus_to_model_text(corpus_file, text_file, tgt_file, topic_file, corpus_type, augmentation, additional_feat, topic_generalization=True):
    """
    convert_conversation_corpus_to_model_text
    """
    fout_text = open(text_file, 'w')
    tgt_text = open(tgt_file, "w")
    fout_topic = open(topic_file, 'w')
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            source, target, topic_dict = preprocessing_for_one_conversation(
                line.strip(), topic_generalization=topic_generalization, augmentation=augmentation, corpus_type=corpus_type)

            topic_dict = json.dumps(topic_dict, ensure_ascii=False)

            if additional_feat:
                for i in range(0, len(source)):
                    source[i] = ltp_pos(source[i])

            if corpus_type == "train":
                for src, tgt in zip(source, target):
                    fout_text.write(src + "\n")
                    tgt_text.write(tgt + "\n")
                    fout_topic.write(topic_dict + "\n")
            else:
                fout_text.write(source[-1] + "\n") # 0-not generalize
                tgt_text.write(target[-1] + "\n")
                fout_topic.write(topic_dict + "\n")

    fout_text.close()
    fout_topic.close()
    tgt_text.close()


def create_qa_dataset(session_file, sample_file_save, text_file_save, tgt_file, topic_file_save, corpus_type):
    assert corpus_type in ["train", "dev", "test"]

    if corpus_type == "!train":
        augmentation = True
    else:
        augmentation = False

    if corpus_type !="test":
        convert_session_to_sample(session_file, sample_file_save)
    else:
        sample_file_save = session_file

    convert_conversation_corpus_to_model_text(sample_file_save, text_file_save, tgt_file, topic_file_save, corpus_type, augmentation=augmentation, additional_feat=False, topic_generalization=True)



def preclean_opt(parse):
    group = parse.add_argument_group("Preclean")
    group.add("--log_file", "-log_file", type=str)
    group.add("--raw_train_file", "-raw_train_file", type=str)
    group.add("--raw_dev_file", "-raw_dev_file", type=str)
    group.add("--raw_test_file", "-raw_test_file", type=str)
    group.add("--train_sample_file_save", "-train_sample_file_save", type=str)
    group.add("--dev_sample_file_save", "-dev_sample_file_save", type=str)
    group.add("--test_sample_file_save", "-test_sample_file_save", type=str)
    group.add("--train_text_file_save", "-train_text_file_save", type=str)
    group.add("--dev_text_file_save", "-dev_text_file_save", type=str)
    group.add("--test_text_file_save", "-test_text_file_save", type=str)
    group.add("--train_topic_file_save", "-train_topic_file_save", type=str)
    group.add("--dev_topic_file_save", "-dev_topic_file_save", type=str)
    group.add("--test_topic_file_save", "-test_topic_file_save", type=str)
    group.add("--train_tgt_file", "-train_tgt_file", type=str)
    group.add("--dev_tgt_file", "-dev_tgt_file", type=str)
    group.add("--test_tgt_file", "-test_tgt_file", type=str)
    return parse


def main(opt):
    create_qa_dataset(opt.raw_train_file, opt.train_sample_file_save, opt.train_text_file_save, opt.train_tgt_file, opt.train_topic_file_save,  "train")
    create_qa_dataset(opt.raw_dev_file, opt.dev_sample_file_save, opt.dev_text_file_save, opt.dev_tgt_file, opt.dev_topic_file_save, "dev")
    create_qa_dataset(opt.raw_test_file, opt.test_sample_file_save, opt.test_text_file_save, opt.test_tgt_file, opt.test_topic_file_save, "test")


if __name__ == '__main__':
    # example = {"goal": [["START", "托马斯 · 桑斯特", "陈思宇"], ["托马斯 · 桑斯特", "出生 日期", "1990 - 5 - 16"], ["陈思宇", "出生 日期", "1990 - 5 - 16"]], "knowledge": [["托马斯 · 桑斯特", "血型", "A型"], ["托马斯 · 桑斯特", "标签", "口碑 很好"], ["托马斯 · 桑斯特", "获奖", "移动迷宫_提名 _ ( 2015 ； 第17届 ) _ 青少年选择奖 _ 青少年选择奖 - 最佳 电影 火花"], ["托马斯 · 桑斯特", "性别", "男"], ["托马斯 · 桑斯特", "职业", "演员"], ["托马斯 · 桑斯特", "领域", "明星"], ["托马斯 · 桑斯特", "星座", "金牛座"], ["陈思宇", "星座", "金牛座"], ["陈思宇", "毕业 院校", "北京电影学院"], ["陈思宇", "体重", "65kg"], ["陈思宇", "性别", "男"], ["陈思宇", "职业", "演员"], ["陈思宇", "领域", "明星"], ["托马斯 · 桑斯特", "评论", "第一次 看到 这 孩子 是 在 《 真爱至上 》 ， 萌 翻 了 ， 现在 长大 了 气质 不错"], ["托马斯 · 桑斯特", "主要成就", "2004年 金卫星奖 年轻 男演员 奖 提名"], ["托马斯 · 桑斯特", "代表作", "神秘博士第三季"]], "conversation": ["知道 外国 有 个 明星 长 得 很 萌 吗 ？", "这个 还 真 不知道 呢 ， 请问 是 谁 啊 ？", "是 托马斯 · 桑斯特 ， 颜值 太 高 了 。", "哦 ， 没 应 说过 呢 ， 你 能 给 大体 说说 么 ？", "给 你 大体 说说 ， 他 口碑 很好 的 ， 也 很 有 才华 ， 我们 国家 有 个 小 哥哥 跟 他 一样 都是 1990年5月16日 出生 的 。", "是 谁 啊 ？", "陈思宇 ， 金牛座 的 ， 毕业 于 北京电影学院 。", "有 时间 了解 一下 。"]}
    # single_example_process(example)
    parse = configargparse.ArgumentParser(default_config_files=[CONFIG_FILE],
                                          config_file_parser_class=configargparse.YAMLConfigFileParser)
    opt = preclean_opt(parse).parse_args()
    init_logger(opt.log_file)
    main(opt)






