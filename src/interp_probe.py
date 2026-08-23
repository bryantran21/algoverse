"""Per-layer linear probes on the residual stream.

NEGATIVE RESULT -- see README finding 9
---------------------------------------
This was planned as the test of whether the answer is linearly *present* at
depths where the logit lens cannot read it out. It was abandoned, and the reason
is worth preserving.

At the 5pt operating point the failure set is 239 yes / 14 no. With 14 items in
the minority class:

- accuracy is meaningless (a majority-class guesser scores 0.95);
- AUROC is computed from pairs, so the whole statistic is determined by where 14
  items rank among 239 -- one item moving flips it substantially;
- the best score obtained (0.847 at layer 9) was an isolated spike in an
  otherwise flat curve, i.e. what noise looks like.

The imbalance itself turned out to be the more informative finding: image-mode
errors are 89% false "no", which is a directional response bias, not random
confusion.

The code is kept because the design is sound and would work on a class-balanced
failure set -- which needs either a balanced dataset or a much larger N.
`check_viable` refuses to run below a minimum minority-class count.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_MINORITY = 40  # below this, probe results are not reportable


def check_viable(y, min_minority: int = MIN_MINORITY) -> tuple[bool, str]:
    """Refuse to report probe results on a set too imbalanced to support them."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    minority = min(n_pos, n_neg)
    if minority < min_minority:
        return False, (f"minority class n={minority} < {min_minority}; "
                       f"AUROC would rest on {minority} items and is not reportable "
                       f"(majority baseline = {max(n_pos, n_neg)/len(y):.3f})")
    return True, f"n_pos={n_pos} n_neg={n_neg}"


def _clf(C: float = 0.1):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C, class_weight="balanced"),
    )


def probe_layers(H_train, y_train, H_test, y_test, C: float = 0.1, cv: int = 5) -> dict:
    """Train one probe per layer on `H_train`, evaluate on `H_test`.

    H_* are [n_items, n_layers, hidden]. Uses AUROC rather than accuracy because
    accuracy is uninformative under class imbalance.
    """
    n_layers = H_train.shape[1]
    train_cv, test_auc, test_bal = [], [], []
    for L in range(n_layers):
        clf = _clf(C)
        train_cv.append(cross_val_score(clf, H_train[:, L], y_train,
                                        cv=cv, scoring="roc_auc").mean())
        clf.fit(H_train[:, L], y_train)
        test_auc.append(roc_auc_score(y_test, clf.predict_proba(H_test[:, L])[:, 1]))
        test_bal.append(balanced_accuracy_score(y_test, clf.predict(H_test[:, L])))
    return {
        "train_cv_auc": np.array(train_cv),
        "test_auc": np.array(test_auc),
        "test_balanced_acc": np.array(test_bal),
    }


def run(H_train, y_train, H_test, y_test, strict: bool = True, **kw) -> dict | None:
    """Probe with a viability gate. Returns None (and explains) if not viable."""
    ok, msg = check_viable(y_test)
    print(f"test set: {msg}")
    if not ok:
        print("SKIPPING probe -- results would not be reportable.")
        if strict:
            return None
    return probe_layers(H_train, y_train, H_test, y_test, **kw)
