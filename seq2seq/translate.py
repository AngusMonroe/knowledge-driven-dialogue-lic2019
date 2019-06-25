import configargparse
from onmt.translate.translator import build_translator
import sys
import onmt.opts as opts
from onmt.utils.logging import init_logger, logger
from eval import calc_bleu, calc_f1, calc_distinct


CONFIG_FILE = "./config/translator.yml"


def main(opt):
    translator = build_translator(opt, report_score=True)
    translator.translate(
        src=opt.src,
        tgt=opt.tgt,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        default_config_files=[CONFIG_FILE],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args()
    init_logger("outputs/log/search_parameters.log")
    main(opt)

    sent_file = []
    with open("./data/all.tgt", 'r') as tgt_fr, open("outputs/pred.txt", 'r') as pred_fr, open("./data/all_topic.txt", 'r') as topic_fr:
        sents = []
        for tgt, pred, topic in zip(tgt_fr, pred_fr, topic_fr):
            topic = eval(topic)
            tgt = tgt.strip("\n")
            tgt = tgt.replace("<SOS> ", "")
            tgt = tgt.replace(" <EOS>", "")

            pred = pred.strip("\n").split("\t")[0]
            # print(pred)
            pred = pred.replace("<SOS> ", "")
            pred = pred.replace(" <EOS>", "")

            for k, v in topic.items():
                tgt = tgt.replace(k, v)
                pred = pred.replace(k, v)
            pred_tokens = pred.strip().split(" ")
            gold_tokens = tgt.strip().split(" ")
            sents.append([pred_tokens, gold_tokens])
        # calc f1
        f1 = calc_f1(sents)
        # calc bleu
        bleu1, bleu2 = calc_bleu(sents)
        # calc distinct
        distinct1, distinct2 = calc_distinct(sents)
        score = f1 + bleu1 + bleu2
        output_str = "Score: %.2f%%\n" % (score * 100)
        output_str += "F1: %.2f%%\n" % (f1 * 100)
        output_str += "BLEU1: %.3f%%\n" % bleu1
        output_str += "BLEU2: %.3f%%\n" % bleu2
        output_str += "DISTINCT1: %.3f%%\n" % distinct1
        output_str += "DISTINCT2: %.3f%%\n" % distinct2
        logger.info(output_str)