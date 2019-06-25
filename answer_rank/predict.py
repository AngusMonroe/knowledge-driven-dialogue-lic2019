import codecs
import configargparse
import pandas as pd
from utils.pkl_util import load_pkl


def predict(opt):
    # ------------------ load data & model ---------------------------
    if opt.predict_type == "dev":
        X = load_pkl(opt.train_feature)
        raw_text = pd.read_csv(opt.train_raw_text, sep="\t")
    else:
        X = load_pkl(opt.test_feature)
        raw_text = pd.read_csv(opt.test_raw_text, sep= "\t")
    print("Shape of X is: %s" %str(X.shape))
    print("Shape of text is: %s"%str(raw_text.shape))
    model = load_pkl(opt.model_checkpoint)

    # ------------------- predict & convert into text ------------------
    y_pred = model.predict(X)
    y_pred = y_pred.reshape(-1, opt.ensemble_model_num)

    best_index = y_pred.argmax(axis=1).tolist()
    candidates = raw_text["preds"].values.reshape(-1, opt.ensemble_model_num).tolist()

    fw = codecs.open(opt.final_pred_save, "w")
    for ix, candidate in zip(best_index, candidates):
        fw.write(candidate[ix])
    fw.close()
    print("Please check %s for final predict text file." % opt.final_pred_save)


def predict_opt(parser):
    parser.add_argument("--predict_type", "-predict_type", type=str, choices=["dev", "test"], default="test")
    parser.add_argument("--train_feature", "-train_feature", type=str, default="features/train/X_10.pkl")
    parser.add_argument("--test_feature", "-test_feature", type=str, default="features/test/X_10.pkl")
    parser.add_argument("--train_raw_text", "-train_raw_text", type=str, default="data/train_raw.txt",
                        help="raw train data for construct features")
    parser.add_argument("--test_raw_text", "-test_raw_text", type=str, default="data/test_raw.txt",
                        help="raw test data for construct features")
    parser.add_argument("--model_checkpoint", "-model_checkpoint", type=str,
                        default="outputs/checkpoint/dh_pretender_X10.pkl")
    parser.add_argument("--ensemble_model_num", "-ensemble_model_num", type=int, default=10,
                        help=" N seq2seq models for ensemble")
    parser.add_argument("--final_pred_save", "-final_pred_save", type=str, default="outputs/result/pred.txt")


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    predict_opt(parser)
    opt = parser.parse_args()