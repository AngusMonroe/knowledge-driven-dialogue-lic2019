import os
from onmt.utils.logging import init_logger, logger
import warnings
import configargparse
warnings.filterwarnings("ignore")

def yield_parameters():
    alpha = [i * 0.2 for i in range(0, 5)]
    beta = [i * 0.2 for i in range(0, 5)]
    beam_size = [1, 2, 3, 4, 5]
    for a in alpha:
        for b in beta:
            for s in beam_size:
                yield (a, b, s)

def search():
    parameters = yield_parameters()
    for ix, (alpha, beta, beam_size) in enumerate(parameters):
        logger.info("--------------------------------------------")
        logger.info("Turns %d: -alpha %.1f -beta %.1f -beam_size %d"%(ix, alpha, beta, beam_size))
        os.system("python translate.py --alpha %d --beta %d --beam_size %d"%(alpha, beta, beam_size))
        logger.info("--------------------------------------------")

def search_opt(parse):
    group = parse.add_argument_group("Search_parameters")
    group.add("--log_file", "-log_file", type=str, default="outputs/log/search_parameters.log")
    return parse

if __name__ == "__main__":
    parse = configargparse.ArgumentParser()
    opt = search_opt(parse).parse_args()
    init_logger(opt.log_file)
    search()

