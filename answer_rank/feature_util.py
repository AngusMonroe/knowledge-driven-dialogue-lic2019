import pandas as pd
import re
from collections import Counter
from eval import calc_bleu, calc_distinct, calc_f1
import re
import string
import os
import scipy
import numpy as np
from pkl_util import to_pkl, load_pkl
import operator

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
import sys
import string
from collections import defaultdict
# import torch

import numpy as np
import pandas as pd

import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils, pkl_utils
from utils import logging_utils, time_utils
from feature_base import BaseEstimator
from multiprocessing import Pool


def get_goal_knowledge(src):
    goal_knowledge = src.split("[Q=0]")[0]
    goal_knowledge = goal_knowledge.replace("[KG] ", "").replace("[", "")
    goal_knowledge = re.sub("Goal=\d] ","", goal_knowledge)
    return goal_knowledge

def get_conver(src):
    conver = src.split("[Q=0]")[1].replace("[", "")
    conver = re.sub("Q=\d] ", "", conver)
    conver = re.sub("A=\d] ", "", conver).replace("]", "")
    return conver

def get_last_question(src):
    last = src.split("]")[-1]
    return last.strip()


def compute_bleu1(p, s):
    bleu1, bleu2 = calc_bleu([[p.split(), s.split()]])
    return bleu1

def compute_bleu2(p, s):
    bleu1, bleu2 = calc_bleu([[p.split(), s.split()]])
    return bleu2

def compute_f1(p, s):
    f1 = calc_f1([[p.split(), s.split()]])
    return f1

def compute_distinct1(p, s):
    d1, d2 = calc_distinct([[p.split(), s.split()]])
    return d1

def compute_distinct2(p, s):
    d1, d2 = calc_distinct([[p.split(), s.split()]])
    return d2

def is_question_sent(p):
    if "吗" in p or "？" in p:
        return 1
    else:
        return 0

def repeat_word_count(p):
    count = Counter()
    count.update(p.split())
    n = 0
    for k, v in count.items():
        n += v
    return n

def entity_overlap_num(p, gk):
    count = Counter()
    gk = gk.split()
    count.update(gk)
    p = p.split()
    n = 0
    for word in p:
        if count.get(word) is not None:
            n+=1
    return n


def get_goal_knowledge_list(src):
    goal_knowledge = src.split("[Q=0]")[0]
    goal_knowledge = re.sub("=\d] ", "", goal_knowledge)
    goal = goal_knowledge.split("[Goal")[1:]
    entitys = []
    for item in goal:
        if "[KG]" not in item:
            entitys.append(item.strip())
        else:
            entitys.extend([i.strip() for i in item.split("[KG]")])
    return entitys


def get_cooccur_pair(s):
    pairs = []
    s = s.split()
    for i in range(len(s)):
        j = i
        while j < len(s) - 1:
            pairs.append([s[i], s[j]])
            j += 1
    return pairs


def entity_cooccur(pred, src):
    entitys = get_goal_knowledge_list(src)
    n = 0
    for e in entitys:
        pairs = get_cooccur_pair(e)
        for pair in pairs:
            token1, token2 = pair
            if token1 in pred and token2 in pred:
                n += 1
    return n

class VectorSpace:

    ## word based
    def _init_word_bow(self, ngram, vocabulary=None):
        bow = CountVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                # norm="l2",
                                strip_accents="unicode",
                                analyzer="word",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                vocabulary=vocabulary)
        return bow

    ## word based
    def _init_word_ngram_tfidf(self, ngram, vocabulary=None):
        tfidf = TfidfVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                norm="l2",
                                strip_accents="unicode",
                                analyzer="word",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1,
                                # stop_words="english",
                                vocabulary=vocabulary)
        return tfidf

    ## char based
    def _init_char_tfidf(self, include_digit=False):
        chars = list(string.ascii_lowercase)
        if include_digit:
            chars += list(string.digits)
        vocabulary = dict(zip(chars, range(len(chars))))
        tfidf = TfidfVectorizer(strip_accents="unicode",
                                analyzer="char",
                                norm=None,
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, 1),
                                use_idf=0,
                                vocabulary=vocabulary)
        return tfidf

    ## char based ngram
    def _init_char_ngram_tfidf(self, ngram, vocabulary=None):
        tfidf = TfidfVectorizer(min_df=3,
                                max_df=0.75,
                                max_features=None,
                                norm="l2",
                                strip_accents="unicode",
                                analyzer="char",
                                token_pattern=r"\w{1,}",
                                ngram_range=(1, ngram),
                                use_idf=1,
                                smooth_idf=1,
                                sublinear_tf=1,
                                # stop_words="english",
                                vocabulary=vocabulary)
        return tfidf


# LSA
class LSA_Word_Ngram(VectorSpace):
    def __init__(self, obs_corpus, place_holder, ngram=3, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        X = tfidf.fit_transform(self.obs_corpus)
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


class LSA_Char_Ngram(VectorSpace):
    def __init__(self, obs_corpus, place_holder, ngram=5, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Char_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        X = tfidf.fit_transform(self.obs_corpus)
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)

# ------------------------ Cooccurrence LSA -------------------------------
# 1st in CrowdFlower
class LSA_Word_Ngram_Cooc(VectorSpace):
    def __init__(self, obs_corpus, target_corpus,
            obs_ngram=1, target_ngram=1, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.obs_ngram = obs_ngram
        self.target_ngram = target_ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.obs_ngram_str = ngram_utils._ngram_str_map[self.obs_ngram]
        self.target_ngram_str = ngram_utils._ngram_str_map[self.target_ngram]

    def __name__(self):
        return "LSA%d_Word_Obs_%s_Target_%s_Cooc"%(
            self.svd_dim, self.obs_ngram_str, self.target_ngram_str)

    def _get_cooc_terms(self, lst1, lst2, join_str):
        out = [""] * len(lst1) * len(lst2)
        cnt =  0
        for item1 in lst1:
            for item2 in lst2:
                out[cnt] = item1 + join_str + item2
                cnt += 1
        res = " ".join(out)
        return res

    def transform(self):
        # ngrams
        obs_ngrams = list(map(lambda x: ngram_utils._ngrams(x.split(" "), self.obs_ngram, "_"), self.obs_corpus))
        target_ngrams = list(map(lambda x: ngram_utils._ngrams(x.split(" "), self.target_ngram, "_"), self.target_corpus))
        # cooccurrence ngrams
        cooc_terms = list(map(lambda lst1,lst2: self._get_cooc_terms(lst1, lst2, "X"), obs_ngrams, target_ngrams))
        ## tfidf
        tfidf = self._init_word_ngram_tfidf(ngram=1)
        X = tfidf.fit_transform(cooc_terms)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        return svd.fit_transform(X)


# 2nd in CrowdFlower (preprocessing_mikhail.py)
class LSA_Word_Ngram_Pair(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s_Pair"%(self.svd_dim, self.ngram_str)

    def transform(self):
        ## tfidf
        tfidf = self._init_word_ngram_tfidf(ngram=self.ngram)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        X_target = tfidf.fit_transform(self.target_corpus)
        X_tfidf = scipy.sparse.hstack([X_obs, X_target]).tocsr()
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        X_svd = svd.fit_transform(X_tfidf)
        return X_svd


# -------------------------------- TSNE ------------------------------------------
# 2nd in CrowdFlower (preprocessing_mikhail.py)
class TSNE_LSA_Word_Ngram(LSA_Word_Ngram):
    def __init__(self, obs_corpus, place_holder, ngram=3, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, None, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Word_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


class TSNE_LSA_Char_Ngram(LSA_Char_Ngram):
    def __init__(self, obs_corpus, place_holder, ngram=5, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, None, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Char_%s" % (self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


class TSNE_LSA_Word_Ngram_Pair(LSA_Word_Ngram_Pair):
    def __init__(self, obs_corpus, target_corpus, ngram=2, svd_dim=100, svd_n_iter=5):
        super().__init__(obs_corpus, target_corpus, ngram, svd_dim, svd_n_iter)

    def __name__(self):
        return "TSNE_LSA%d_Word_%s_Pair" % (self.svd_dim, self.ngram_str)

    def transform(self):
        X_svd = super().transform()
        X_scaled = StandardScaler().fit_transform(X_svd)
        X_tsne = TSNE().fit_transform(X_scaled)
        return X_tsne


# ------------------------ TFIDF Cosine Similarity -------------------------------
class TFIDF_Word_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=3):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "TFIDF_Word_%s_CosineSim" % self.ngram_str

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class TFIDF_Char_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "TFIDF_Char_%s_CosineSim" % self.ngram_str

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


# ------------------------ LSA Cosine Similarity -------------------------------
class LSA_Word_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=3, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Word_%s_CosineSim" % (self.svd_dim, self.ngram_str)

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_word_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_word_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        svd.fit(scipy.sparse.vstack((X_obs, X_target)))
        X_obs = svd.transform(X_obs)
        X_target = svd.transform(X_target)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class LSA_Char_Ngram_CosineSim(VectorSpace):
    def __init__(self, obs_corpus, target_corpus, ngram=5, svd_dim=100, svd_n_iter=5):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus
        self.ngram = ngram
        self.svd_dim = svd_dim
        self.svd_n_iter = svd_n_iter
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "LSA%d_Char_%s_CosineSim" % (self.svd_dim, self.ngram_str)

    def transform(self):
        ## get common vocabulary
        tfidf = self._init_char_ngram_tfidf(self.ngram)
        tfidf.fit(list(self.obs_corpus) + list(self.target_corpus))
        vocabulary = tfidf.vocabulary_
        ## obs tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_obs = tfidf.fit_transform(self.obs_corpus)
        ## targetument tfidf
        tfidf = self._init_char_ngram_tfidf(self.ngram, vocabulary)
        X_target = tfidf.fit_transform(self.target_corpus)
        ## svd
        svd = TruncatedSVD(n_components=self.svd_dim,
                           n_iter=self.svd_n_iter, random_state=config.RANDOM_SEED)
        svd.fit(scipy.sparse.vstack((X_obs, X_target)))
        X_obs = svd.transform(X_obs)
        X_target = svd.transform(X_target)
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


# ------------------- Char Distribution Based features ------------------
# 2nd in CrowdFlower (preprocessing_stanislav.py)
class CharDistribution(VectorSpace):
    def __init__(self, obs_corpus, target_corpus):
        self.obs_corpus = obs_corpus
        self.target_corpus = target_corpus

    def normalize(self, text):
        # pat = re.compile("[a-z0-9]")
        pat = re.compile("[a-z]")
        group = pat.findall(text.lower())
        if group is None:
            res = " "
        else:
            res = "".join(group)
            res += " "
        return res

    def preprocess(self, corpus):
        return [self.normalize(text) for text in corpus]

    def get_distribution(self):
        ## obs tfidf
        tfidf = self._init_char_tfidf()
        X_obs = tfidf.fit_transform(self.preprocess(self.obs_corpus)).todense()
        X_obs = np.asarray(X_obs)
        # apply laplacian smoothing
        s = 1.
        X_obs = (X_obs + s) / (np.sum(X_obs, axis=1)[:, None] + X_obs.shape[1] * s)
        ## targetument tfidf
        tfidf = self._init_char_tfidf()
        X_target = tfidf.fit_transform(self.preprocess(self.target_corpus)).todense()
        X_target = np.asarray(X_target)
        X_target = (X_target + s) / (np.sum(X_target, axis=1)[:, None] + X_target.shape[1] * s)
        return X_obs, X_target


class CharDistribution_Ratio(CharDistribution):
    def __init__(self, obs_corpus, target_corpus, const_A=1., const_B=1.):
        super().__init__(obs_corpus, target_corpus)
        self.const_A = const_A
        self.const_B = const_B

    def __name__(self):
        return "CharDistribution_Ratio"

    def transform(self):
        X_obs, X_target = self.get_distribution()
        ratio = (X_obs + self.const_A) / (X_target + self.const_B)
        return ratio


class CharDistribution_CosineSim(CharDistribution):
    def __init__(self, obs_corpus, target_corpus):
        super().__init__(obs_corpus, target_corpus)

    def __name__(self):
        return "CharDistribution_CosineSim"

    def transform(self):
        X_obs, X_target = self.get_distribution()
        ## cosine similarity
        sim = list(map(dist_utils._cosine_sim, X_obs, X_target))
        sim = np.asarray(sim).squeeze()
        return sim


class CharDistribution_KL(CharDistribution):
    def __init__(self, obs_corpus, target_corpus):
        super().__init__(obs_corpus, target_corpus)

    def __name__(self):
        return "CharDistribution_KL"

    def transform(self):
        X_obs, X_target = self.get_distribution()
        ## KL
        kl = dist_utils._KL(X_obs, X_target)
        return kl


token_pattern = " "  # just split the text into tokens


# ----------------------------- TF ------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocTF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocTF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocTF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(s)
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ----------------------------- Normalized TF ------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocNormTF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocNormTF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocNormTF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(np_utils._try_divide(s, len(target_ngrams)))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------------ TFIDF -----------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocTFIDF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5) / (self.df_dict[word] + 0.5))

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(s * self._get_idf(w1))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------------ Normalized TFIDF -----------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocNormTFIDF_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocNormTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocNormTFIDF_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5) / (self.df_dict[word] + 0.5))

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            val_list.append(np_utils._try_divide(s, len(target_ngrams)) * self._get_idf(w1))
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list


# ------------------------ BM25 ---------------------------------------------
# StatCooc stands for StatisticalCooccurrence
class StatCoocBM25_Ngram(BaseEstimator):
    """Single aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode="",
                 str_match_threshold=config.STR_MATCH_THRESHOLD, k1=config.BM25_K1, b=config.BM25_B):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.k1 = k1
        self.b = b
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]
        self.str_match_threshold = str_match_threshold
        self.df_dict = self._get_df_dict()
        self.avg_ngram_doc_len = self._get_avg_ngram_doc_len()

    def __name__(self):
        if isinstance(self.aggregation_mode, str):
            feat_name = "StatCoocBM25_%s_%s" % (
                self.ngram_str, string.capwords(self.aggregation_mode))
        elif isinstance(self.aggregation_mode, list):
            feat_name = ["StatCoocBM25_%s_%s" % (
                self.ngram_str, string.capwords(m)) for m in self.aggregation_mode]
        return feat_name

    def _get_df_dict(self):
        # smoothing
        d = defaultdict(lambda: 1)
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            for w in set(target_ngrams):
                d[w] += 1
        return d

    def _get_idf(self, word):
        return np.log((self.N - self.df_dict[word] + 0.5) / (self.df_dict[word] + 0.5))

    def _get_avg_ngram_doc_len(self):
        lst = []
        for target in self.target_corpus:
            target_tokens = nlp_utils._tokenize(target, token_pattern)
            target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
            lst.append(len(target_ngrams))
        return np.mean(lst)

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        K = self.k1 * (1 - self.b + self.b * np_utils._try_divide(len(target_ngrams), self.avg_ngram_doc_len))
        val_list = []
        for w1 in obs_ngrams:
            s = 0.
            for w2 in target_ngrams:
                if dist_utils._is_str_match(w1, w2, self.str_match_threshold):
                    s += 1.
            bm25 = s * self._get_idf(w1) * np_utils._try_divide(1 + self.k1, s + K)
            val_list.append(bm25)
        if len(val_list) == 0:
            val_list = [config.MISSING_VALUE_NUMERIC]
        return val_list

def max_word_freq(s):
    s = s.split()
    count = Counter()
    count.update(s)
    sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_count[0][1]

def mean_word_freq(s):
    s = s.split()
    count = Counter()
    count.update(s)
    n = 0
    for k, v in count.items():
        n +=v
    return n/len(count)


# df features
def run_df_feature(raw_data):
    df = pd.read_csv(raw_data, sep="\t")
    print(df.head())
    df["pred_word_length"] = df["preds"].apply(lambda s: len(s.split()))
    df["pred_char_length"] = df["preds"].apply(lambda s: len(s.replace(" ", "")))
    df["goal_and_knowledge"] = df["src"].apply(lambda s: get_goal_knowledge(s))
    df["conver"] = df["src"].apply(lambda s: get_conver(s))
    df["last_conver"] = df["src"].apply(lambda s: get_last_question(s))
    df["bleu1_pred_src"] = df.apply(lambda row: compute_bleu1(row["preds"], row["src"]), axis=1)
    df["bleu2_pred_src"] = df.apply(lambda row: compute_bleu2(row["preds"], row["src"]), axis=1)
    df["bleu1_pred_conver"] = df.apply(lambda row: compute_bleu1(row["preds"], row["conver"]), axis=1)
    df["bleu2_pred_conver"] = df.apply(lambda row: compute_bleu2(row["preds"], row["conver"]), axis=1)
    df["bleu1_pred_last_conver"] = df.apply(lambda row: compute_bleu1(row["preds"], row["last_conver"]), axis=1)
    df["bleu2_pred_last_conver"] = df.apply(lambda row: compute_bleu2(row["preds"], row["last_conver"]), axis=1)
    df["pred_is_question_sent"] = df["preds"].apply(lambda s: is_question_sent(s))
    df["last_conver_is_question_sent"] = df["last_conver"].apply(lambda s: is_question_sent(s))
    df["pred_repeat_word_cnt"] = df["preds"].apply(lambda s: repeat_word_count(s))
    df["entity_overlap_num"] = df.apply(lambda row: entity_overlap_num(row["preds"], row["goal_and_knowledge"]), axis=1)
    df["f1_pred_src"] = df.apply(lambda row: compute_f1(row["preds"], row["src"]), axis=1)
    df["f1_pred_conver"] = df.apply(lambda row: compute_f1(row["preds"], row["conver"]), axis=1)
    df["f1_pred_last_conver"] = df.apply(lambda row: compute_f1(row["preds"], row["last_conver"]), axis=1)
    df["distinct1_pred_src"] = df.apply(lambda row: compute_distinct1(row["preds"], row["src"]), axis=1)
    df["distinct2_pred_src"] = df.apply(lambda row: compute_distinct2(row["preds"], row["src"]), axis=1)
    df["distinct1_pred_conver"] = df.apply(lambda row: compute_distinct1(row["preds"], row["conver"]), axis=1)
    df["distinct2_pred_conver"] = df.apply(lambda row: compute_distinct2(row["preds"], row["conver"]), axis=1)
    df["distinct1_pred_last_conver"] = df.apply(lambda row: compute_distinct1(row["preds"], row["last_conver"]), axis=1)
    df["distinct2_pred_last_conver"] = df.apply(lambda row: compute_distinct2(row["preds"], row["last_conver"]), axis=1)
    df["pred_gk_cooccur"] = df.apply(lambda row: entity_cooccur(row["preds"], row["src"]), axis=1)
    df["max_word_freq"] = df["preds"].apply(lambda s: max_word_freq(s))
    df["mean_word_freq"] = df["preds"].apply(lambda s: mean_word_freq(s))
    print("shape of df is:", df.shape)
    return df


# LSA n gram
def run_lsa_ngram(df, field):
    obj_corpus = df[field].values
    n_grams = [1, 2, 3]
    for n_gram in n_grams:
        ext = LSA_Word_Ngram(obj_corpus, None, n_gram, config.SVD_DIM, config.SVD_N_ITER)
        x = ext.transform()
        save_path = "features/feature_lsa_word_%d_gram_%s.pkl"%(n_gram, field)
        to_pkl(x, save_path)

def run_lsa_char_ngram(df, field):
    obj_corpus = df[field].values
    n_grams = [1, 2, 3, 4]

    for n_gram in n_grams:
        ext = LSA_Char_Ngram(obj_corpus, None, n_gram, config.SVD_DIM, config.SVD_N_ITER)
        x = ext.transform()
        save_path = "features/feature_lsa_char_%d_gram_%s.pkl"%(n_gram, field)
        to_pkl(x, save_path)

# LSA n gram cosinesim
def run_lsa_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = LSA_Word_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_lsa_cosinesim_word_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

def run_lsa_char_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = LSA_Char_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_lsa_cosinesim_char_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

# tfidf sim
def run_tfidf_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = TFIDF_Word_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_tfidf_cosinesim_word_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

def run_tfidf_char_ngram_cosinesim(obj_field, target_field):
    n_grams = [1, 2, 3]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for n_gram in n_grams:
        ext = TFIDF_Char_Ngram_CosineSim(obj_corpus, tgt_corpus, n_gram)
        x = ext.transform()
        print(x.shape)
        save_path = "features/feature_tfidf_cosinesim_char_%d_gram_%s_%s.pkl"%(n_gram, obj_field, target_field)
        to_pkl(x, save_path)

# dist sim
def run_char_dist_sim(obj_field, target_field, generator):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%s_%s_%s.pkl"%(ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)

# tsne lsa ngram
def run_lsa_ngram_cooc(obj_field, target_field, generator):
    obs_ngrams = [1, 2]
    target_ngrams = [1, 2]
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    for obs_ngram in obs_ngrams:
        for target_ngram in target_ngrams:
            ext = generator(obj_corpus, tgt_corpus, obs_ngram=obs_ngram, target_ngram=target_ngram)
            x = ext.transform()
            print(x.shape)
            save_path = "features/feature_%s_%s_%s.pkl"%(ext.__name__(), obj_field, target_field)
            to_pkl(x, save_path)

def run_tfidf(obj_field, target_field, generator, n_gram):
    obj_corpus = df[obj_field].values
    tgt_corpus = df[target_field].values
    ext = generator(obj_corpus, tgt_corpus, ngram=n_gram)
    x = ext.transform()
    print(x.shape)
    save_path = "features/feature_%d_gram_%s_%s_%s.pkl"%(n_gram, ext.__name__(), obj_field, target_field)
    to_pkl(x, save_path)


def dump_df_feature(df, fields):
    for field in fields:
        data = df[field].values
        save_path = "features/feature_%s.pkl" % (field)
        to_pkl(data, save_path)
        # torch.save(data, save_path)

def dumps_y(df):
    y = df["score"].values
    save_path = "features/train/y_10.pkl"
    to_pkl(y, save_path)
    # torch.save(y, save_path)

def feature_combine(feature_dir):
    features = []
    file_names = os.listdir(feature_dir)
    for file_name in file_names:
        if not file_name.startswith("feature"):
            continue
        feature = load_pkl(os.path.join(feature_dir, file_name))
        if len(feature.shape) == 1:
            feature = feature[np.newaxis, :].transpose()
        features.append(feature)
    print("features", len(features))
    X = np.concatenate(features, axis=1)
    print("X shape is:", X.shape)
    to_pkl(X, "features/train/X_10.pkl")


if __name__ == '__main__':
    df = run_df_feature(config.raw_data)
    run_lsa_ngram(df, "preds")
    run_lsa_ngram(df, "src")

    run_lsa_char_ngram(df, "preds")
    run_lsa_char_ngram(df, "src")
    #
    obj_fields = ["preds"]
    target_fields = ["src", "goal_and_knowledge", "conver", "last_conver"]
    for obj_field in obj_fields:
        for target_field in target_fields:
            run_lsa_ngram_cosinesim(obj_field, target_field)

    for obj_field in obj_fields:
        for target_field in target_fields:
            run_lsa_char_ngram_cosinesim(obj_field, target_field)

    for obj_field in obj_fields:
        for target_field in target_fields:
            run_tfidf_ngram_cosinesim(obj_field, target_field)

    for obj_field in obj_fields:
        for target_field in target_fields:
            run_tfidf_char_ngram_cosinesim(obj_field, target_field)

    generators = [CharDistribution_Ratio, CharDistribution_CosineSim, CharDistribution_KL]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                run_char_dist_sim(obj_field, target_field, generator)

    generators = [LSA_Word_Ngram_Cooc]
    for obj_field in obj_fields:
        for target_field in target_fields:
            for generator in generators:
                run_lsa_ngram_cooc(obj_field, target_field, generator)

    # generators = [StatCoocTF_Ngram, StatCoocNormTF_Ngram, StatCoocTFIDF_Ngram, StatCoocNormTFIDF_Ngram,
    #               StatCoocBM25_Ngram]
    # n_grams = [1, 2]
    # for obj_field in obj_fields:
    #     for target_field in target_fields:
    #         for generator in generators:
    #             for n_gram in n_grams:
    #                 run_tfidf(obj_field, target_field, generator, n_gram)

    fields = ["beam_score","model_weight", "max_word_freq", "mean_word_freq", 'pred_word_length', 'pred_char_length', 'bleu1_pred_src',
              'bleu2_pred_src', 'bleu1_pred_conver', 'bleu2_pred_conver',
              'bleu1_pred_last_conver', 'bleu2_pred_last_conver',
              'pred_is_question_sent', 'entity_overlap_num', 'f1_pred_src',
              'f1_pred_conver', 'f1_pred_last_conver', 'distinct1_pred_src',
              'distinct2_pred_src', 'distinct1_pred_conver', 'distinct2_pred_conver',
              'distinct1_pred_last_conver', 'distinct2_pred_last_conver',
              'pred_gk_cooccur', 'last_conver_is_question_sent',
              'pred_repeat_word_cnt']
    dump_df_feature(df, fields)


    if config.is_training:
        features = feature_combine("./features/")
        dumps_y(df)
    else:
        features = feature_combine("./features/")








