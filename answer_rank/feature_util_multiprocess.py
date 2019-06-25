from collections import Counter
from eval import calc_bleu, calc_distinct, calc_f1
import re
import os
import scipy
from utils.pkl_util import to_pkl, load_pkl
import operator

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import config
from utils import dist_utils, ngram_utils, nlp_utils, np_utils
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
                self.ngram_str, string.capw