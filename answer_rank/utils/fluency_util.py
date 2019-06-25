from collections import Counter
from answer_rank.utils.ngram_utils import _unigrams, _bigrams, _trigrams

def get_repeat_num(count):
    repeat = 0
    for k, v in count.items():
        if v > 1:
            repeat += v
    return repeat

def evaluate_fluency(sentence):
    words = sentence.split()
    unigram_words = _unigrams(words)
    bigrams_words = _bigrams(words, "_")
    trigrams_words = _trigrams(words, "_")
    uni_count = Counter(unigram_words)
    bi_count = Counter(bigrams_words)
    tri_count = Counter(trigrams_words)

    uni_repeat = get_repeat_num(uni_count)
    bi_repeat = get_repeat_num(bi_count)
    tri_count = get_repeat_num(tri_count)
    repeat = uni_repeat + bi_repeat + tri_count
    if repeat == 0:
        fluency = 1
    else:
        fluency = 1 / repeat
    return fluency

if __name__ == '__main__':
    s1 = "网友 评论 说 知道 为什么 他 没 特里尔 名气 大 吗 ， 因为 他 的 嘴 不大 ， 因为 他 的 嘴 不大 。"
    s2 = "网友 评论 说 知道 为什么 他 没 特里尔 名气 大 吗 ， 因为 他 的 嘴 不大 。"
    f1 = evaluate_fluency(s1)
    f2 = evaluate_fluency(s2)
    print(f1, f2)
