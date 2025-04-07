import torch

class ClassificationMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def forward(self, y_pred, y_true):
        y_pred = (y_pred > self.threshold).bool()
        y_true = y_true.bool()
        tp = (y_pred & y_true).sum().item()
        tn = (~y_pred & ~y_true).sum().item()
        fp = (y_pred & ~y_true).sum().item()
        fn = (~y_pred & y_true).sum().item()
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
        return self._cls_metrics(tp, tn, fp, fn)

    def reset(self):
        self.tp = self.tn = self.fp = self.fn = 0

    def _cls_metrics(self, tp, tn, fp, fn):
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return accuracy, precision, recall, f1

    def epoch_metrics(self):
        metrics = self._cls_metrics(self.tp, self.tn, self.fp, self.fn)
        self.reset()
        return metrics
