from copy import deepcopy

import numpy as np
import torch as t
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from torch import Tensor

from CCS import CCS
from CRC import CRC


def fit_logreg(
        train_pos: Float[Tensor, "batch d_hidden"],
        train_neg: Float[Tensor, "batch d_hidden"],
        train_labels: np.ndarray,

) -> float:
    lr = LogisticRegression(max_iter=10000)
    lr.fit(train_pos-train_neg, train_labels)
    return lr

def fit_ccs(pos, neg, labels, normalize, n_probes=50, device=t.device("cuda")):
    p = deepcopy(pos)
    n = deepcopy(neg)
    
    ccs = CCS(
        pos=p,
        neg=n,
        normalize=normalize,
        n_probe=n_probes,
        device=device
    )
    ccs.optimize()
    
    accs = []
    # for probe in ccs.probes:
    #     preds = ccs.predict(probe, pos, neg)
    #     acc = (preds == labels).mean()
    #     acc = max(acc, 1-acc)
    #     accs.append(acc)
    return accs, deepcopy(ccs)

def fit_crc(pos, neg, normalize):
    p = deepcopy(pos)
    n = deepcopy(neg)
    
    crc = CRC()
    crc.fit(p, n, normalize)
    return crc
    