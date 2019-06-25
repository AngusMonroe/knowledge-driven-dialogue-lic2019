import os
import sys

import numpy as np

import answer_rank.config as config


class BaseEstimator:
    def __init__(self, obs_corpus, target_corpus, aggregation_mode, id_list=None, aggregation_mode_prev=""):
        self.obs_corpus = obs_corpus
        self.N = len(obs_corpus)
        # for standalone feature, we use the same interface, so better take care of it
        self.target_corpus = range(self.N) if target_corpus is None else target_corpus
        # id_list is used for group based relevance/distance features
        self.id_list = range(self.N) if id_list is None else id_list
        # aggregation for list features, e.g., intersect positions
        self.aggregation_mode, self.aggregator = self._check_aggregation_mode(aggregation_mode)
        self.aggregation_mode_prev, self.aggregator_prev = self._check_aggregation_mode(aggregation_mode_prev)
        self.double_aggregation = False
        if self.aggregator_prev != [None]:
            # the output of transform_one is a list of list, i.e., [[...], [...], [...]]
            self.double_aggregation = True

    def _check_aggregation_mode(self, aggregation_mode):
        valid_aggregation_modes = ["", "size", "mean", "std", "max", "min", "median"]
        if isinstance(aggregation_mode, str):
            assert aggregation_mode.lower() in valid_aggregation_modes, "Wrong aggregation_mode: %s"%aggregation_mode
            aggregation_mode = [aggregation_mode.lower()]
        elif isinstance(aggregation_mode, list):
            for m in aggregation_mode:
                assert m.lower() in valid_aggregation_modes, "Wrong aggregation_mode: %s"%m
            aggregation_mode = [m.lower() for m in aggregation_mode]

        aggregator = [None if m == "" else getattr(np, m) for m in aggregation_mode]

        return aggregation_mode, aggregator

    def transform(self):
        # original score
        score = list(map(self.transform_one, self.obs_corpus, self.target_corpus, self.id_list))
        # aggregation
        if isinstance(score[0], list):
            if self.double_aggregation:
                # double aggregation
                res = np.zeros((self.N, len(self.aggregator_prev) * len(self.aggregator)), dtype=float)
                for m,aggregator_prev in enumerate(self.aggregator_prev):
                    for n,aggregator in enumerate(self.aggregator):
                        idx = m * len(self.aggregator) + n
                        for i in range(self.N):
                            # process in a safer way
                            try:
                                tmp = []
                                for l in score[i]:
                                    try:
                                        s = aggregator_prev(l)
                                    except:
                                        s = config.MISSING_VALUE_NUMERIC
                                    tmp.append(s)
                            except:
                                tmp = [ config.MISSING_VALUE_NUMERIC ]
                            try:
                                s = aggregator(tmp)
                            except:
                                s = config.MISSING_VALUE_NUMERIC
                            res[i,idx] = s
            else:
                # single aggregation
                res = np.zeros((self.N, len(self.aggregator)), dtype=float)
                for m,aggregator in enumerate(self.aggregator):
                    for i in range(self.N):
                        # process in a safer way
                        try:
                            s = aggregator(score[i])
                        except:
                            s = config.MISSING_VALUE_NUMERIC
                        res[i,m] = s
        else:
            res = np.asarray(score, dtype=float)
        return res
