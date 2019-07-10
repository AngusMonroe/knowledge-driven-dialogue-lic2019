import os
import platform

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils import os_utils


# ---------------------- Overall -----------------------
TASK = "all"
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = "./"

DATA_DIR = "%s/Data"%ROOT_DIR
CLEAN_DATA_DIR = "%s/Clean"%DATA_DIR

FEAT_DIR = "%s/Feat"%ROOT_DIR
FEAT_FILE_SUFFIX = ".pkl"
FEAT_CONF_DIR = "./conf"

OUTPUT_DIR = "%s/Output"%ROOT_DIR
SUBM_DIR = "%s/Subm"%OUTPUT_DIR

LOG_DIR = "%s/Log"%ROOT_DIR
FIG_DIR = "%s/Fig"%ROOT_DIR
TMP_DIR = "%s/Tmp"%ROOT_DIR
THIRDPARTY_DIR = "%s/Thirdparty"%ROOT_DIR

# word2vec/doc2vec/glove
WORD2VEC_MODEL_DIR = "%s/word2vec"%DATA_DIR
GLOVE_WORD2VEC_MODEL_DIR = "%s/glove/gensim"%DATA_DIR
DOC2VEC_MODEL_DIR = "%s/doc2vec"%DATA_DIR

# index split
SPLIT_DIR = "%s/split"%DATA_DIR

# dictionary
WORD_REPLACER_DATA = "%s/dict/word_replacer.csv"%DATA_DIR

# colors
COLOR_DATA = "%s/dict/color_data.py"%DATA_DIR

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = "%s/train.csv"%DATA_DIR
TEST_DATA = "%s/test.csv"%DATA_DIR
ATTR_DATA = "%s/attributes.csv"%DATA_DIR
DESC_DATA = "%s/product_descriptions.csv"%DATA_DIR
SAMPLE_DATA = "%s/sample_submission.csv"%DATA_DIR

ALL_DATA_RAW = "%s/all.raw.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED = "%s/all.lemmatized.csv.pkl"%CLEAN_DATA_DIR
ALL_DATA_LEMMATIZED_STEMMED = "%s/all.lemmatized.stemmed.csv.pkl"%CLEAN_DATA_DIR
INFO_DATA = "%s/info.csv.pkl"%CLEAN_DATA_DIR

# size
TRAIN_SIZE = 74067
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 166693
VALID_SIZE_MAX = 60000 # 0.7 * TRAIN_SIZE

TRAIN_MEAN = 2.381634
TRAIN_VAR = 0.285135

TEST_MEAN = TRAIN_MEAN
TEST_VAR = TRAIN_VAR

MEAN_STD_DICT = {
    1.00: 0.000, # Common: [1, 1, 1]
    1.25: 0.433, # Rare: [1,1,1,2]
    1.33: 0.471, # Common: [1, 1, 2]
    1.50: 0.866, # Rare: [1, 1, 1, 3]
    1.67: 0.471, # Common: [1, 2, 2]
    1.75: 0.829, # Rare: [1, 1, 2, 3]
    2.00: 0.000, # Common: [2, 2, 2], [1, 2, 3]
    2.25: 0.829, # Rare: [1,2,3,3]
    2.33: 0.471, # Common: [2, 2, 3]
    2.50: 0.500, # Rare: [2,2,3,3]
    2.67: 0.471, # Common: [2, 3, 3]
    2.75: 0.433, # Rare: [2,3,3,3]
    3.00: 0.000, # Common: [3, 3, 3]
}

# ------------------------ PARAM ------------------------
# attribute name and value SEPARATOR
ATTR_SEPARATOR = " | "

# cv
N_RUNS = 5
N_FOLDS = 1

# intersect count/match
STR_MATCH_THRESHOLD = 0.85

# correct query with google spelling check dict
# turn this on/off to have two versions of features/models
# which is useful for ensembling
GOOGLE_CORRECTING_QUERY = True

# auto correcting query (quite time consuming; not used in final submission)
AUTO_CORRECTING_QUERY = False

# query expansion (not used in final submission)
QUERY_EXPANSION = False

# bm25
BM25_K1 = 1.6
BM25_B = 0.75

# svd
SVD_DIM = 100
SVD_N_ITER = 5

# xgboost
# mean of relevance in training set
BASE_SCORE = TRAIN_MEAN

# word2vec/doc2vec
EMBEDDING_ALPHA = 0.025
EMBEDDING_LEARNING_RATE_DECAY = 0.5
EMBEDDING_N_EPOCH = 5
EMBEDDING_MIN_COUNT = 3
EMBEDDING_DIM = 100
EMBEDDING_WINDOW = 5
EMBEDDING_WORKERS = 6

# count transformer
COUNT_TRANSFORM = np.log1p

# missing value
MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.

# stop words
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# ------------------------ OTHER ------------------------
RANDOM_SEED = 2016
PLATFORM = platform.system()
NUM_CORES = 4 if PLATFORM == "Windows" else 14

DATA_PROCESSOR_N_JOBS = 4 if PLATFORM == "Windows" else 6
AUTO_SPELLING_CHECKER_N_JOBS = 4 if PLATFORM == "Windows" else 8


## rgf
RGF_CALL_EXE = "%s/rgf1.2/test/call_exe.pl"%THIRDPARTY_DIR
RGF_EXTENSION = ".exe" if PLATFORM == "Windows" else ""
RGF_EXE = "%s/rgf1.2/bin/rgf%s"%(THIRDPARTY_DIR, RGF_EXTENSION)


# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CLEAN_DATA_DIR]
DIRS += [SPLIT_DIR]
DIRS += [FEAT_DIR, FEAT_CONF_DIR]
DIRS += ["%s/All"%FEAT_DIR]
DIRS += ["%s/Run%d"%(FEAT_DIR,i+1) for i in range(N_RUNS)]
DIRS += ["%s/Combine"%FEAT_DIR]
DIRS += [OUTPUT_DIR, SUBM_DIR]
DIRS += ["%s/All"%OUTPUT_DIR]
DIRS += ["%s/Run%d"%(OUTPUT_DIR,i+1) for i in range(N_RUNS)]
DIRS += [LOG_DIR, FIG_DIR, TMP_DIR]
DIRS += [WORD2VEC_MODEL_DIR, DOC2VEC_MODEL_DIR, GLOVE_WORD2VEC_MODEL_DIR]

# os_utils._create_dirs(DIRS)

train_raw_data = "data/train_raw.txt"
test_raw_data = "data/test_raw.txt"
is_training = True
model_save_path = "outputs/xgbranker-3.model"
train_file = "features/train/X.pkl"
target_file = "features/train/y.pkl"
test_file = "features/train/X.pkl"
predict_save = "outputs/y_pred.txt"
test_sample_file = "data/samples.json"
predict_index_save = "outputs/predict_index_save.pkl"

# model_weight_dict = {'2019-5-1-17w-rnn-dev.txt': 337,
#  '2019-5-1-17w-rnn-layer2-dev.txt': 294,
#  '2019-5-1-17w-rnn-tencent-dev.txt': 317,
#  '2019-5-1-17w-transformer-dev.txt': 557,
#  '2019-5-1-36w-rnn-dev.txt': 354,
#  '2019-5-1-36w-rnn-layer2-dev.txt': 477,
#  '2019-5-1-36w-rnn-tencent-dev.txt': 565,
#  '2019-5-1-36w-transformer-dev.txt': 483,
#  '2019-5-1-46w-rnn-dev.txt': 1056,
#  '2019-5-1-46w-rnn-layer2-dev.txt': 673,
#  '2019-5-1-46w-rnn-tencent-dev.txt': 337,
#  '2019-5-1-46w-transformer-dev.txt': 612,
#  '2019-5-1-60w-rnn-dev.txt': 430,
#  '2019-5-1-60w-rnn-layer2-dev.txt': 319,
#  '2019-5-1-60w-rnn-tencent-dev.txt': 528,
#  '2019-5-1-60w-transformer-dev.txt': 742,
#  '2019-5-1-8w-rnn-dev.txt': 260,
#  '2019-5-1-8w-rnn-layer2-dev.txt': 484,
#  '2019-5-1-8w-rnn-tencent-dev.txt': 229}

# model_weight_dict = {'2019-5-1-17w-layer31-dev.txt': 157,
#  '2019-5-1-17w-rnn-dev.txt': 257,
#  '2019-5-1-17w-rnn-layer2-dev.txt': 205,
#  '2019-5-1-17w-rnn-tencent-dev.txt': 231,
#  '2019-5-1-17w-transformer-dev.txt': 419,
#  '2019-5-1-36w-rnn-dev.txt': 249,
#  '2019-5-1-36w-rnn-layer2-dev.txt': 374,
#  '2019-5-1-36w-rnn-tencent-dev.txt': 455,
#  '2019-5-1-36w-transfomer-dev.txt': 361,
#  '2019-5-1-40w-layer2-dev.txt': 123,
#  '2019-5-1-40w-rnn-dev.txt': 246,
#  '2019-5-1-40w-rnn-layer31-dev.txt': 178,
#  '2019-5-1-40w-rnn-tencent-dev.txt': 354,
#  '2019-5-1-40w-transformer-dev.txt': 564,
#  '2019-5-1-46w-rnn-dev.txt': 846,
#  '2019-5-1-46w-rnn-layer2-dev.txt': 544,
#  '2019-5-1-46w-rnn-layer31-dev.txt': 278,
#  '2019-5-1-46w-rnn-tencent-dev.txt': 246,
#  '2019-5-1-46w-transformer-dev.txt': 471,
#  '2019-5-1-60w-rnn-dev.txt': 314,
#  '2019-5-1-60w-rnn-layer2-dev.txt': 235,
#  '2019-5-1-60w-rnn-tencent-dev.txt': 422,
#  '2019-5-1-60w-transformer-dev.txt': 538,
#  '2019-5-1-8w-rnn-dev.txt': 206,
#  '2019-5-1-8w-rnn-layer2-dev.txt': 394,
#  '2019-5-1-8w-rnn-layer31-dev.txt': 228,
#  '2019-5-1-8w-rnn-tencent-dev.txt': 159}

model_weight_dict = {'2019-5-1-17w-rnn-layer31-test.txt': 157,
 '2019-5-1-17w-rnn-test.txt': 257,
 '2019-5-1-17w-rnn-layer2-test.txt': 205,
 '2019-5-1-17w-rnn-tencent-test.txt': 231,
 '2019-5-1-17w-transformer-test.txt': 419,
 '2019-5-1-36w-rnn-test.txt': 249,
 '2019-5-1-36w-rnn-layer2-test.txt': 374,
 '2019-5-1-36w-rnn-tencent-test.txt': 455,
 '2019-5-1-36w-transformer-test.txt': 361,
 '2019-5-1-40w-rnn-layer2-test.txt': 123,
 '2019-5-1-40w-rnn-test.txt': 246,
 '2019-5-1-40w-rnn-layer31-test.txt': 178,
 '2019-5-1-40w-rnn-tencent-test.txt': 354,
 '2019-5-1-40w-transformer-test.txt': 564,
 '2019-5-1-46w-rnn-test.txt': 846,
 '2019-5-1-46w-rnn-layer2-test.txt': 544,
 '2019-5-1-46w-rnn-layer31-test.txt': 278,
 '2019-5-1-46w-rnn-tencent-test.txt': 246,
 '2019-5-1-46w-transformer-test.txt': 471,
 '2019-5-1-60w-rnn-test.txt': 314,
 '2019-5-1-60w-rnn-layer2-test.txt': 235,
 '2019-5-1-60w-rnn-tencent-test.txt': 422,
 '2019-5-1-60w-transformer-test.txt': 538,
 '2019-5-1-8w-rnn-test.txt': 206,
 '2019-5-1-8w-rnn-layer2-test.txt': 394,
 '2019-5-1-8w-rnn-layer31-test.txt': 228,
 '2019-5-1-8w-rnn-tencent-test.txt': 159}

fields = ["model_weight", "max_word_freq", "mean_word_freq",
              'pred_word_length', 'pred_char_length', 'bleu1_pred_src',
              'bleu2_pred_src', 'bleu1_pred_conver', 'bleu2_pred_conver',
              'bleu1_pred_last_conver', 'bleu2_pred_last_conver',
              'pred_is_question_sent', 'entity_overlap_num', 'f1_pred_src',
              'f1_pred_conver', 'f1_pred_last_conver', 'distinct1_pred_src',
              'distinct2_pred_src', 'distinct1_pred_conver', 'distinct2_pred_conver',
              'distinct1_pred_last_conver', 'distinct2_pred_last_conver',
              'pred_gk_cooccur', 'last_conver_is_question_sent',
              'pred_repeat_word_cnt']

# model_weight_dict = {'2019-5-1-17w-rnn-test.txt': 337,
#  '2019-5-1-17w-rnn-layer2-test.txt': 294,
#  '2019-5-1-17w-rnn-tencent-test.txt': 317,
#  '2019-5-1-17w-transformer-test.txt': 557,
#  '2019-5-1-36w-rnn-test.txt': 354,
#  '2019-5-1-36w-rnn-layer2-test.txt': 477,
#  '2019-5-1-36w-rnn-tencent-test.txt': 565,
#  '2019-5-1-36w-transformer-test.txt': 483,
#  '2019-5-1-46w-rnn-test.txt': 1056,
#  '2019-5-1-46w-rnn-layer2-test.txt': 673,
#  '2019-5-1-46w-rnn-tencent-test.txt': 337,
#  '2019-5-1-46w-transformer-test.txt': 612,
#  '2019-5-1-60w-rnn-test.txt': 430,
#  '2019-5-1-60w-rnn-layer2-test.txt': 319,
#  '2019-5-1-60w-rnn-tencent-test.txt': 528,
#  '2019-5-1-60w-transformer-test.txt': 742,
#  '2019-5-1-8w-rnn-test.txt': 260,
#  '2019-5-1-8w-rnn-layer2-test.txt': 484,
#  '2019-5-1-8w-rnn-tencent-test.txt': 229}

# model_weight_dict ={'2019-5-1-17w-rnn-test.txt': 619,
#  '2019-5-1-17w-rnn-tencent-test.txt': 605,
#  '2019-5-1-36w-rnn-test.txt': 772,
#  '2019-5-1-36w-rnn-tencent-test.txt': 1004,
#  '2019-5-1-36w-transformer-test.txt': 936,
#  '2019-5-1-46w-rnn-test.txt': 1577,
#  '2019-5-1-46w-transformer-test.txt': 974,
#  '2019-5-1-60w-rnn-test.txt': 804,
#  '2019-5-1-60w-transformer-test.txt': 1170,
#  '2019-5-1-8w-rnn-test.txt': 593}

# model_weight_dict ={'2019-5-1-17w-rnn-dev.txt': 619,
#  '2019-5-1-17w-rnn-tencent-dev.txt': 605,
#  '2019-5-1-36w-rnn-dev.txt': 772,
#  '2019-5-1-36w-rnn-tencent-dev.txt': 1004,
#  '2019-5-1-36w-transfomer-dev.txt': 936,
#  '2019-5-1-46w-rnn-dev.txt': 1577,
#  '2019-5-1-46w-transformer-dev.txt': 974,
#  '2019-5-1-60w-rnn-dev.txt': 804,
#  '2019-5-1-60w-transformer-dev.txt': 1170,
#  '2019-5-1-8w-rnn-dev.txt': 593}