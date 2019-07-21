import numpy as np


def _spoof_metric(pred: np.ndarray, labels: np.ndarray, threshold: float):
    pred = (pred > threshold).astype('float32')
    labels = (labels != 3).astype('float32')
    tp = np.sum(pred * labels)
    tn = np.sum((1 - pred) * (1 - labels))
    fp = np.sum(pred * (1 - labels))
    fn = np.sum((1 - pred) * labels)

    metric = fp / (fp + tn) + 19 * fn / (fn + tp)
    return metric


def spoof_metric(pred: np.ndarray, labels: np.ndarray):
    pred = pred[:, :-1].sum(axis=1)
    acc = [_spoof_metric(pred, labels, threshold=x) for x in pred]
    return min(acc)


def accuracy(pred: np.ndarray, labels: np.ndarray):
    pred = pred.argmax(axis=1)
    return (pred == labels).mean()
