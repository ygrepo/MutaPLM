import numpy as np
from scipy.stats import spearmanr

def loss(y_true, y_pred):
    return np.mean(y_pred)

def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).statistic

name2metric = {
    "spearmanr": spearman,
    "loss": loss,
}