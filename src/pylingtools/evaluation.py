from collections import namedtuple


class EvalResultsBinary(object):
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        
    def add(self, evalRes):
        self.tp += evalRes.tp
        self.fp += evalRes.fp
        self.fn += evalRes.fn
        
    def __str__(self):
        return "tp={} fp={} fn={}".format(self.tp, self.fp, self.fn)


def calc_precision_recall_fbeta(evalResBin, beta = 1.):
    print "Tp:", evalResBin.tp
    print "Tp + Fp:", evalResBin.tp + evalResBin.fp
    precision = (1. * evalResBin.tp) / (evalResBin.tp + evalResBin.fp)
    recall = (1. * evalResBin.tp) / (evalResBin.tp + evalResBin.fn)
    fbeta = 2. * precision * recall / (precision + recall)
    
    stats = namedtuple("Stats", "precision, recall, fbeta")
    
    return stats(precision, recall, fbeta)
