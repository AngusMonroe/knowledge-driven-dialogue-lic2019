import os
import json
from tqdm import tqdm
import configargparse

from onmt.utils.logging import init_logger, logger
from pyltp import Postagger, NamedEntityRecognizer

CONFIG_FILE = "./config/preclean.yml"

postagger = Postagger()
postagger.load("ltp_model/pos.model")


def ltp_pos(inputs):
    inputs = inputs.split()
    pos_inputs = []
    postags = list(postagger.postag(inputs))
    for d, p in zip(inputs, postags):
        if d.startswith("[Session-") or d=="[CLS]" or d=="[SEP]" or d.startswith("[Goal-") or d=="[K-G]":
            p = "special"
        pos_inputs.append(d + "|" + p)
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
        input_from_goal.append("[Goal-%d]" %ix)
        for token in g:
            input_from_goal.append(token)
    return input_from_goal


def knowledge_process(knowledge):
    input_from_knowledge = []
    for k in knowledge:
        input_from_knowledge.append("[K-G]")
        for token in k:
            input_from_knowledge.append(token)
    # input_from_knowledge.append("[SEP]")
    return input_from_knowledge


def conversation_process(conversation):
    input_from_conversation = []
    target_from_conversation = []
    turns = 1
    dialogue = []
    dialogue.append("[Session-0] [CLS]")
    for index, conver in enumerate(conversation):
        if index % 2 == 1:
            dialogue.append("[Session-%d] "%turns + conver)
            turns += 1
        else:
            dialogue.append(conver)

    # print("dialogue", dialogue)

    ix = 0
    while ix < len(dialogue):
        if ix % 2 == 1:
            target_from_conversation.append("<SOS> " + dialogue[ix] + " <EOS>")
            input_from_conversation.append((dialogue[max(0, ix-5): ix], dialogue[max(0, ix-3): ix])) # 保留前两轮对话

        ix += 1
    return input_from_conversation, target_from_conversation


def single_example_process(example):
    goal = example["goal"]
    knowledge = example["knowledge"]
    conversation = example["conversation"]
    input_from_goal = goal_process(goal)
    # input_from_knowledge = knowledge_process(knowledge)
    input_from_conversation, target_from_conversation = conversation_process(conversation)
    src, tgt = [], []
    for conver, target in zip(input_from_conversation, target_from_conversation):
        for c in conver:
            valid_knowledge = knowledge_select(target, knowledge)
            input_from_knowledge = knowledge_process(valid_knowledge)
            inputs = " ".join(c) + " " + " ".join(input_from_knowledge) + " " + " ".join(input_from_goal)
            # inputs = ltp_pos(inputs)
            src.append(inputs)
            tgt.append(target)
    # for s, t in zip(src, tgt):
    #     print("A: ", s)
    #     print("B: ", t)
    return src, tgt


def writer(src, tgt, src_fw, tgt_fw):
    for s, g in zip(src, tgt):
        src_fw.write(s + "\n")
        tgt_fw.write(g + "\n")


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
        print(kg)
        _kg = " ".join(kg)
        overlap_num = _compute_overlap_num(tgt, _kg)
        if overlap_num > 0:
            valid_knowledge.append(kg)
    return valid_knowledge


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
    src_fw = open(os.path.join(save_dir, "%s.src" % corpus_type), "w")
    tgt_fw = open(os.path.join(save_dir, "%s.tgt" % corpus_type), "w")
    examples = make_examples(raw_data_path)
    for example in tqdm(examples):
        src, tgt = single_example_process(example)
        src, tgt = drop_duplicate(src, tgt)
        writer(src, tgt, src_fw, tgt_fw)
    logger.info("%s process done!" % corpus_type)


def history_process(history):
    dialogue = []
    if len(history) == 0:
        dialogue.append("[Session-0] [CLS]")
    else:
        turns = 1
        dialogue = []
        dialogue.append("[Session-0] [CLS]")
        for index, conver in enumerate(history):
            if index % 2 == 1:
                dialogue.append("[Session-%d] " % turns + conver)
                turns += 1
            else:
                dialogue.append(conver)
    return dialogue


def test_input_select(history, knowledge, goal):
    new_knowledge = []
    new_goal = []
    # print(history)
    if history == ["[Session-0] [CLS]"]:
        # print(goal)
        new_goal.append(goal[0])
        for kg in knowledge:
            kg = get_words(kg)
            _goal = get_words(goal[0])
            overlap_num = _compute_overlap_num(_goal, kg)
            if overlap_num > 0:
                new_knowledge.append(kg)
    else:
        new_goal = goal
        for h in history[-2:]:
            for kg in knowledge:
                kg = get_words(kg)
                overlap_num = _compute_overlap_num(h, kg)
                if overlap_num > 0:
                    new_knowledge.append(kg)

        # # 使用goal作为kg的选取参考
        # new_goal = goal
        # # 首先统计目标goal和在历史对话中是否出现
        # # 选出出现过的goal，用他来选取kg
        # _goal = []
        # for h in history:
        #     for g in goal:
        #         overlap_num = _compute_overlap_num(h, g)
        #         if overlap_num > 0:
        #             _goal.append(goal)
        # if len(_goal) > 0:
        #     for g in _goal:
        #         for kg in knowledge:
        #             overlap_num = _compute_overlap_num(g, kg)
        #             if overlap_num > 0:
        #                 new_knowledge.append(kg)
        # else:
        #     # 如果goal都没出现过的话，那么拿第一个goal去选取kg
        #     new_goal = [goal[0]]
        #     for kg in knowledge:
        #         overlap_num = _compute_overlap_num(goal[0], kg)
        #         if overlap_num > 0:
        #             new_knowledge.append(kg)
    return new_knowledge, new_goal


def single_test_process(example):
    goal = example["goal"]
    knowledge = example["knowledge"]
    history = example["history"]
    input_from_history = history_process(history)
    new_knowledge, new_goal = test_input_select(input_from_history, knowledge, goal)
    new_knowledge = knowledge_process(new_knowledge)
    new_goal = goal_process(new_goal)
    src = " ".join(input_from_history) + " " + " ".join(new_knowledge) + " " + " ".join(new_goal)
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
    group.add("--log_file", "-log_file", type=str)
    group.add("--raw_train_file", "-raw_train_file", type=str)
    group.add("--raw_dev_file", "-raw_dev_file", type=str)
    group.add("--save_dir", "-save_dir", type=str)
    group.add("--raw_test_file", "-raw_test_file", type=str)
    return parse


def main(opt):
    # create_qa_dataset(opt.raw_train_file, opt.save_dir, "train")
    # create_qa_dataset(opt.raw_dev_file, opt.save_dir, "dev")
    create_test_dataset(opt.raw_test_file, opt.save_dir, "test")


if __name__ == '__main__':
    # example = {"goal": [["START", "托马斯 · 桑斯特", "陈思宇"], ["托马斯 · 桑斯特", "出生 日期", "1990 - 5 - 16"], ["陈思宇", "出生 日期", "1990 - 5 - 16"]], "knowledge": [["托马斯 · 桑斯特", "血型", "A型"], ["托马斯 · 桑斯特", "标签", "口碑 很好"], ["托马斯 · 桑斯特", "获奖", "移动迷宫_提名 _ ( 2015 ； 第17届 ) _ 青少年选择奖 _ 青少年选择奖 - 最佳 电影 火花"], ["托马斯 · 桑斯特", "性别", "男"], ["托马斯 · 桑斯特", "职业", "演员"], ["托马斯 · 桑斯特", "领域", "明星"], ["托马斯 · 桑斯特", "星座", "金牛座"], ["陈思宇", "星座", "金牛座"], ["陈思宇", "毕业 院校", "北京电影学院"], ["陈思宇", "体重", "65kg"], ["陈思宇", "性别", "男"], ["陈思宇", "职业", "演员"], ["陈思宇", "领域", "明星"], ["托马斯 · 桑斯特", "评论", "第一次 看到 这 孩子 是 在 《 真爱至上 》 ， 萌 翻 了 ， 现在 长大 了 气质 不错"], ["托马斯 · 桑斯特", "主要成就", "2004年 金卫星奖 年轻 男演员 奖 提名"], ["托马斯 · 桑斯特", "代表作", "神秘博士第三季"]], "conversation": ["知道 外国 有 个 明星 长 得 很 萌 吗 ？", "这个 还 真 不知道 呢 ， 请问 是 谁 啊 ？", "是 托马斯 · 桑斯特 ， 颜值 太 高 了 。", "哦 ， 没 应 说过 呢 ， 你 能 给 大体 说说 么 ？", "给 你 大体 说说 ， 他 口碑 很好 的 ， 也 很 有 才华 ， 我们 国家 有 个 小 哥哥 跟 他 一样 都是 1990年5月16日 出生 的 。", "是 谁 啊 ？", "陈思宇 ， 金牛座 的 ， 毕业 于 北京电影学院 。", "有 时间 了解 一下 。"]}
    # single_example_process(example)
    parse = configargparse.ArgumentParser(default_config_files=[CONFIG_FILE],
                                          config_file_parser_class=configargparse.YAMLConfigFileParser)
    opt = preclean_opt(parse).parse_args()
    init_logger(opt.log_file)
    main(opt)






