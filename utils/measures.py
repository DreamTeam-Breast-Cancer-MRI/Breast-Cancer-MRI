import numpy as np


def tp(true_labels, pred_labels) -> int:
    return np.sum(np.logical_and(pred_labels == 1, true_labels == 1))


def tn(true_labels, pred_labels) -> int:
    return np.sum(np.logical_and(pred_labels == 0, true_labels == 0))


def fp(true_labels, pred_labels) -> int:
    return np.sum(np.logical_and(pred_labels == 1, true_labels == 0))


def fn(true_labels, pred_labels) -> int:
    return np.sum(np.logical_and(pred_labels == 0, true_labels == 1))


def recall(labels, outputs) -> float:
    tp_int = tp(labels, outputs)
    fp_int = fp(labels, outputs)
    return tp_int / (tp_int + fp_int)


def precission(labels, outputs) -> float:
    tp_int = tp(labels, outputs)
    fn_int = fn(labels, outputs)
    return tp_int / (tp_int + fn_int)


def F1(labels, outputs) -> float:
    prec = precission(labels, outputs)
    rec = recall(labels, outputs)

    return 2 * prec * rec / (prec + rec)
