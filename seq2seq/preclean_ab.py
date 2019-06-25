import os
import json
from tqdm import tqdm
import configargparse

from onmt.utils.logging import init_logger, logger
from pyltp import Postagger, NamedEntityRecognizer

CONFIG_FILE = "./config/preclean.yml"

postagger = Postagger()
postagger.load("ltp_model/pos.model")
recognizer = NamedEntityRecognizer()
recognizer.load("ltp_model/ner.model")


def ltp_pos(inputs):
    inputs = inputs.split()
    pos_inputs = []
    postags = list(postagger.postag(inputs))
    netags = list(recognizer.recognize(inputs, postags))
    for d, p, n in zip(inputs, postags, netags):
        if d.startswith("[Session-") or d=="[CLS]" or d=="[SEP]" or d.startswith("[Goal-") or d=="[K-G]":
            p = "special"
            n = "O"
        pos_inputs.append(d + "|" + p + "|" +n)
    # print(pos_inputs)

    return " ".join(pos_inputs)



def make_examples(path):
    with open(path, 'r') as fr:
        for line in fr:
            data = json.loads(line.strip("\n"))
            yield data


def goal_process(goal):
    input_from_goal = []
    for ix, g in enumerate(goal):
        input_from_goal.append("[Goal=%d]" %ix)
        for token in g:
            input_from_goal.append(token)
    return input_from_goal


def knowledge_process(knowledge):
    input_from_knowledge = []
    for k in knowledge:
        input_from_knowledge.append("[KG]")
        for token in k:
            input_from_knowledge.append(token)
    # input_from_knowledge.append("[SEP]")
    return input_from_knowledge


def conversation_process(conversation):
    input_from_conversation = []
    target_from_conversation = []
    turns = 1
    dialogue = []
    dialogue.append("[Q=0] [CLS]")
    for index, conver in enumerate(conversation):
        if index % 2 == 1:
            dialogue.append("[Q=%d] "%turns + conver)
            turns += 1
        else:
            if index == 0:
                a_turns = 0
            else:
                a_turns = turns - 1
            dialogue.append("[A=%d] "%a_turns + conver)


    # print("dialogue", dialogue)

    ix = 0
    while ix < len(dialogue):
        if ix % 2 == 1:
            target_from_conversation.append("<SOS> " + " ".join(dialogue[ix].split()[1:]) + " <EOS>")
            input_from_conversation.append((dialogue[max(0, ix-5): ix], dialogue[max(0, ix-3): ix], dialogue[max(0, ix-1): ix]))   # 保留前两轮对话

        ix += 1
    return input_from_conversation, target_from_conversation


def single_example_process(example):
    goal = example["goal"]
    knowledge = example["knowledge"]
    conversation = example["conversation"]

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

    input_from_goal = goal_process(goal)
    # input_from_knowledge = knowledge_process(knowledge)
    input_from_conversation, target_from_conversation = conversation_process(conversation)
    src, tgt = [], []
    for conver, target in zip(input_from_conversation, target_from_conversation):
        for c in conver:
            valid_knowledge = knowledge_select(target, knowledge)
            input_from_knowledge = knowledge_process(valid_knowledge)
            inputs =  " ".join(input_from_goal)+ " " + " ".join(input_from_knowledge) + " " + " ".join(c)
            # inputs = ltp_pos(inputs)
            src.append(inputs)
            tgt.append(target)
    # for s, t in zip(src, tgt):
    #     print("A: ", s)
    #     print("B: ", t)
    return src, tgt, topic_dict


def writer(src, tgt, src_fw, tgt_fw, topic_fw, topic_dict, generalize):
    for s, g in zip(src, tgt):
        src_fw.write(s + "\n")
        tgt_fw.write(g + "\n")
    if generalize:
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for s, g in zip(src, tgt):
            for key, value in topic_list:
                s = s.replace(value, key)
                g = g.replace(value, key)
            src_fw.write(s + "\n")
            tgt_fw.write(g + "\n")
        topic_dict = json.dumps(topic_dict, ensure_ascii=False)
        topic_fw.write(topic_dict + "\n")


def _compute_overlap_num(tgt, kg):
    if isinstance(tgt, str):
        tgt = list(set(tgt.split()))
    if isinstance(kg, str):
        kg = list(set(kg.split()))
    overlap_num = 0
    for t in tgt:
        for k in kg:
            if t == k:
                overlap_num +=1
    return overlap_num

def get_words(kg):
    words = []
    for item in kg:
        for w in item.split():
            words.append(w)
    return words


def knowledge_select(tgt, knowledge):
    valid_knowledge = []
    for kg in knowledge:
        kg = get_words(kg)
        _kg = " ".join(kg)
        overlap_num = _compute_overlap_num(tgt, _kg)
        if overlap_num > 0:
            valid_knowledge.append(kg)
    return valid_knowledge
    # valid_knowledge = knowledge
    # return valid_knowledge


def goal_select():
    pass


def drop_duplicate(src, tgt):
    src_dict = {}
    new_src = []
    new_tgt=  []
    for s, t in zip(src, tgt):
        if src_dict.get(s) is None:
            new_src.append(s)
            new_tgt.append(t)
            src_dict[s] = 1
    return new_src, new_tgt


def create_qa_dataset(raw_data_path, save_dir, corpus_type):
    logger.info("Start process %s"%raw_data_path)
    src_fw = open(os.path.join(save_dir, "%s_2_or_3_turns.src" % corpus_type), "w")
    tgt_fw = open(os.path.join(save_dir, "%s_2_or_3_turns.tgt" % corpus_type), "w")
    topic_fw = open(os.path.join(save_dir, "%s_2_or_3_turns_topic.txt" % corpus_type), "w")
    examples = make_examples(raw_data_path)
    for example in tqdm(examples):
        src, tgt, topic_dict = single_example_process(example)
        src, tgt = drop_duplicate(src, tgt)
        writer(src, tgt, src_fw, tgt_fw, topic_fw, topic_dict, generalize=True)
    logger.info("%s process done!" % corpus_type)
    src_fw.close()
    tgt_fw.close()
    topic_fw.close()


def history_process(history):
    dialogue = []
    if len(history) == 0:
        dialogue.append("[A=0] [CLS]")
    else:
        turns = 1
        dialogue.append("[Q=0] [CLS]")
        for index, conver in enumerate(history):
            if index % 2 == 1:
                dialogue.append("[Q=%d] " % turns + conver)
                turns += 1
            else:
                if index == 0:
                    a_turns = 0
                else:
                    a_turns = turns - 1
                dialogue.append("[A=%d] " % a_turns + conver)
    return dialogue


def test_input_select(history, knowledge, goal):
    new_knowledge = []
    new_goal = []
    if history == ["[A=0] [CLS]"]:
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


def single_test_process(example):
    goal = example["goal"]
    knowledge = example["knowledge"]
    history = example["conversation"]


    input_from_history = history_process(history)
    new_knowledge, new_goal = test_input_select(input_from_history, knowledge, goal)
    new_knowledge = knowledge_process(new_knowledge)
    new_goal = goal_process(new_goal)
    src = " ".join(new_goal) + " " + " ".join(new_knowledge) + " " + " ".join(input_from_history)
    # src = ltp_pos(src)
    return src


def create_test_dataset(raw_data_path, save_dir, corpus_type):
    src_fw = open(os.path.join(save_dir, "%s.src" % corpus_type), "w")
    examples = make_examples(raw_data_path)
    for example in tqdm(examples):
        src = single_test_process(example)
        src_fw.write(src + '\n')
    src_fw.close()



def preclean_opt(parse):
    group = parse.add_argument_group("Preclean")
    group.add("--log_file", "-log_file", type=str, default="outputs/log/log.txt")
    group.add("--raw_train_file", "-raw_train_file", type=str, default="data/train.txt")
    # group.add("--raw_dev_file", "-raw_dev_file", type=str, default="data/dev.txt")
    group.add("--save_dir", "-save_dir", type=str, default="data/")
    # group.add("--raw_test_file", "-raw_test_file", type=str)
    return parse


def main(opt):
    create_qa_dataset(opt.raw_train_file, opt.save_dir, "train")
    # create_test_dataset(opt.raw_dev_file, opt.save_dir, "dev")
    # create_test_dataset(opt.raw_test_file, opt.save_dir, "test")


if __name__ == '__main__':
    # example = {"goal": [["START", "托马斯 · 桑斯特", "陈思宇"], ["托马斯 · 桑斯特", "出生 日期", "1990 - 5 - 16"], ["陈思宇", "出生 日期", "1990 - 5 - 16"]], "knowledge": [["托马斯 · 桑斯特", "血型", "A型"], ["托马斯 · 桑斯特", "标签", "口碑 很好"], ["托马斯 · 桑斯特", "获奖", "移动迷宫_提名 _ ( 2015 ； 第17届 ) _ 青少年选择奖 _ 青少年选择奖 - 最佳 电影 火花"], ["托马斯 · 桑斯特", "性别", "男"], ["托马斯 · 桑斯特", "职业", "演员"], ["托马斯 · 桑斯特", "领域", "明星"], ["托马斯 · 桑斯特", "星座", "金牛座"], ["陈思宇", "星座", "金牛座"], ["陈思宇", "毕业 院校", "北京电影学院"], ["陈思宇", "体重", "65kg"], ["陈思宇", "性别", "男"], ["陈思宇", "职业", "演员"], ["陈思宇", "领域", "明星"], ["托马斯 · 桑斯特", "评论", "第一次 看到 这 孩子 是 在 《 真爱至上 》 ， 萌 翻 了 ， 现在 长大 了 气质 不错"], ["托马斯 · 桑斯特", "主要成就", "2004年 金卫星奖 年轻 男演员 奖 提名"], ["托马斯 · 桑斯特", "代表作", "神秘博士第三季"]], "conversation": ["知道 外国 有 个 明星 长 得 很 萌 吗 ？", "这个 还 真 不知道 呢 ， 请问 是 谁 啊 ？", "是 托马斯 · 桑斯特 ， 颜值 太 高 了 。", "哦 ， 没 应 说过 呢 ， 你 能 给 大体 说说 么 ？", "给 你 大体 说说 ， 他 口碑 很好 的 ， 也 很 有 才华 ， 我们 国家 有 个 小 哥哥 跟 他 一样 都是 1990年5月16日 出生 的 。", "是 谁 啊 ？", "陈思宇 ， 金牛座 的 ， 毕业 于 北京电影学院 。", "有 时间 了解 一下 。"]}
    # single_example_process(example)
    parse = configargparse.ArgumentParser()
    opt = preclean_opt(parse).parse_args()
    init_logger(opt.log_file)
    main(opt)






