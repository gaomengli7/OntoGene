from typing import Sequence, Callable, Dict
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from seqeval.metrics import accuracy_score
from transformers import EvalPrediction
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
# promotercore任务
def accuracy_score_remote(y_true, y_pred):
    pred_idx = np.argmax(y_pred, axis=1)
    # for y_t, y_p in zip(y_true, pred_idx):
    #     print(y_t, y_p)
    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, pred_idx))
    nb_true = len(y_true)
    score_top1 = nb_correct / nb_true
    return score_top1


def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


def compute_accuracy_metrics(task_name, preds, labels):
    if task_name == 'promotercore':
        accuracy = accuracy_score_remote(labels, preds)
        # precision = precision_score(labels, np.argmax(preds, axis=1), average='weighted')
        recall = recall_score(labels, np.argmax(preds, axis=1), average='weighted')
        f1 = f1_score(labels, np.argmax(preds, axis=1), average='weighted')
        return {
            "accuracy": accuracy_score_remote(labels, preds),
            # "precision": precision,
            "recall": recall,
            "f1":f1,
            "probabilities": preds
        }
    else:
        raise KeyError(task_name)


def simple_accuracy(preds, labels):
    return (preds == labels).float().mean()


def build_compute_metrics_fn(task_name: str, output_type: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_type == 'token-level-classification':
            logits = p.predictions
            preds = np.argmax(logits, axis=-1)
            label_ids = torch.from_numpy(p.label_ids)
            preds = torch.from_numpy(preds)
            active_index = (label_ids.view(-1) != -100)
            active_preds = preds.view(-1)[active_index]
            active_labels = label_ids.view(-1)[active_index]
            return compute_metrics_mapping[task_name](task_name, active_preds, active_labels)
        elif output_type == 'sequence-level-classification' or output_type == 'sequence-level-regression':
            logits = p.predictions
            # preds = np.argmax(logits, axis=1)
            label_ids = p.label_ids

            if task_name == 'promotercore':
                probabilities = p.predictions
                fpr, tpr, _ = roc_curve(label_ids, probabilities[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('假正率')
                plt.ylabel('真正率')
                plt.title('受试者工作特征 (ROC) 曲线')
                plt.legend(loc='lower right')
                plt.show()
            return compute_metrics_mapping[task_name](task_name, logits, label_ids)
        else:
            raise Exception("output type not supported.")

    return compute_metrics_fn


compute_metrics_mapping = {
    'promotercore': compute_accuracy_metrics,

}
