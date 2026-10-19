"""Exploratory editorial-link prediction with nested, purged poem splits.

The target is explicit editorial linkage, not a comprehensive semantic truth.
All transformations estimated from this corpus are fitted within training data.
"""
from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import stylometric_features

CS = (.001, .01, .1, 1., 10., 100., 1000.)
VERSIONS = ("prevalence", "conventional", "plus_input", "plus_fixed_depths", "plus_all_blocks")


def purged_training_indices(pairs, candidates, test, components):
    """Remove both exact poem and declared linked-component overlap."""
    pairs = np.asarray(pairs, int)
    forbidden = {components[int(p)] for p in pairs[np.asarray(test)].ravel()}
    return np.asarray([int(i) for i in candidates
                       if all(components[int(p)] not in forbidden for p in pairs[i])], dtype=int)


def metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    return {"log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, probability)) if np.any(y) else None}


class Features:
    """Fold-specific conventional features plus immutable pretrained distances."""

    def __init__(self, texts, pairs, minilm, fixed_features, feature_names, tfidf_config):
        self.texts = texts
        self.pairs = np.asarray(pairs, int)
        self.minilm = np.asarray(minilm)
        self.fixed = np.asarray(fixed_features)
        self.names = list(feature_names)
        self.config = dict(tfidf_config)
        self.config["ngram_range"] = tuple(self.config["ngram_range"])
        self.style, self.style_names = stylometric_features(texts)
        self.cache = {}
        a, b = self.pairs.T
        endings = [(re.search(r"[^\w\s]+$", text.rstrip()).group(0)
                    if re.search(r"[^\w\s]+$", text.rstrip()) else "") for text in texts]
        self.elementary = np.column_stack([
            abs(self.style[a, 0] - self.style[b, 0]),
            abs(self.style[a, 1] - self.style[b, 1]),
            abs(self.style[a, 2] - self.style[b, 2]),
            (a + b + 2) / (2 * len(texts)),
            [float(endings[i] == endings[j]) for i, j in self.pairs]])

    def get(self, train, version):
        poems = tuple(sorted(set(self.pairs[np.asarray(train)].ravel().tolist())))
        if poems not in self.cache:
            vectorizer = TfidfVectorizer(**self.config, dtype=np.float64)
            vectorizer.fit([self.texts[i] for i in poems])
            vectors = vectorizer.transform(self.texts)
            if np.any(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel() == 0):
                raise ValueError("A poem has no training-vocabulary features")
            a, b = self.pairs.T
            tfidf_distance = 1 - np.asarray(vectors[a].multiply(vectors[b]).sum(axis=1)).ravel()
            mean, scale = self.style[list(poems)].mean(axis=0), self.style[list(poems)].std(axis=0)
            x = np.divide(self.style - mean, scale, out=np.zeros_like(self.style), where=scale > 0)
            norm = np.linalg.norm(x, axis=1)
            if np.any(norm == 0):
                raise ValueError("Undefined style cosine after training scaling")
            x /= norm[:, None]
            style_distance = 1 - (x[a] * x[b]).sum(axis=1)
            self.cache[poems] = np.column_stack([tfidf_distance, style_distance,
                                                self.minilm[a, b], self.elementary])
        base = self.cache[poems]
        if version == "conventional":
            return base
        if version == "plus_input":
            keep = [i for i, name in enumerate(self.names) if name.endswith("/input")]
        elif version == "plus_fixed_depths":
            keep = [i for i, name in enumerate(self.names) if "/fixed/" in name or name.endswith("/input")]
        elif version == "plus_all_blocks":
            keep = [i for i, name in enumerate(self.names) if "/block/" in name or name.endswith("/input")]
        elif version in ("plus_final_block", "plus_final_normalized", "plus_both_final",
                         "final_plus_fixed_depths", "final_plus_all_blocks", "plus_early_blocks"):
            keep = []
            for i, name in enumerate(self.names):
                selected = name.endswith("/input")
                if version in ("plus_final_block", "plus_both_final"):
                    selected |= name.endswith("/fixed/1.0")
                if version in ("plus_final_normalized", "plus_both_final", "final_plus_fixed_depths", "final_plus_all_blocks"):
                    selected |= name.endswith("/final_normalized")
                if version == "final_plus_fixed_depths":
                    selected |= "/fixed/" in name
                if version == "final_plus_all_blocks":
                    selected |= "/block/" in name
                if version == "plus_early_blocks" and "/block/" in name:
                    selected |= int(name.rsplit(".", 1)[1]) < 5
                if selected:
                    keep.append(i)
        else:
            raise ValueError(version)
        return np.column_stack([base, self.fixed[:, keep]])


def fitted_prediction(x, y, train, test, c):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, solver="lbfgs"))
    model.fit(x[train], y[train])
    if model[-1].n_iter_[0] >= 3000:
        raise RuntimeError("Logistic regression failed to converge")
    return model.predict_proba(x[test])[:, 1]


def nested_diagnostic(features, labels, bands, components, *, versions=VERSIONS):
    labels, bands = np.asarray(labels), np.asarray(bands)
    n = len(labels)
    prediction = {name: np.full(n, np.nan) for name in versions}
    folds = []
    for band in sorted(set(bands)):
        test = np.flatnonzero(bands == band)
        initial = np.flatnonzero(bands != band)
        train = purged_training_indices(features.pairs, initial, test, components)
        if len(np.unique(labels[train])) < 2:
            raise ValueError("Purged training set lacks both target classes")
        details = {"band": int(band), "train_rows": train.tolist(), "test_rows": test.tolist(),
                   "purged_rows": [int(i) for i in sorted(set(initial) - set(train))],
                   "train_positive": int(labels[train].sum()), "test_positive": int(labels[test].sum()),
                   "models": {}}
        prior = float(labels[train].mean())
        prediction["prevalence"][test] = prior
        details["models"]["prevalence"] = {"training_prevalence": prior,
                                                 "test": metrics(labels[test], prediction["prevalence"][test])}
        inner = []
        for inner_train, validation in GroupKFold(n_splits=3).split(train, labels[train], bands[train]):
            validation = train[validation]
            inner_train = purged_training_indices(features.pairs, train[inner_train], validation, components)
            if len(np.unique(labels[inner_train])) < 2:
                raise ValueError("Inner purged training set lacks both classes")
            inner.append((inner_train, validation))
        details["inner_splits"] = [{"train_rows": tr.tolist(), "validation_rows": va.tolist()} for tr, va in inner]
        for name in versions[1:]:
            losses = []
            for c in CS:
                values, weights = [], []
                for tr, va in inner:
                    p = fitted_prediction(features.get(tr, name), labels, tr, va, c)
                    values.append(log_loss(labels[va], p, labels=[0, 1])); weights.append(len(va))
                losses.append(float(np.average(values, weights=weights)))
            c = CS[int(np.argmin(losses))]
            x = features.get(train, name)
            p = fitted_prediction(x, labels, train, test, c)
            prediction[name][test] = p
            details["models"][name] = {"C": c, "inner_log_losses": losses,
                                        "feature_count": x.shape[1], "test": metrics(labels[test], p)}
        folds.append(details)
    if any(not np.isfinite(p).all() for p in prediction.values()):
        raise RuntimeError("Incomplete predictions")
    return {"folds": folds, "metrics": {name: metrics(labels, p) for name, p in prediction.items()},
            "predictions": {name: p.tolist() for name, p in prediction.items()}}


def paired_stability(y, bands, predictions, *, reference="plus_input", repetitions=2000, seed=20261018):
    """Conditional block stability, with all predictions and model fits fixed."""
    y, bands = np.asarray(y), np.asarray(bands)
    p = {k: np.asarray(v) for k, v in predictions.items()}
    units = sorted(set(bands)); rng = np.random.default_rng(seed)
    rows = []
    for name in predictions:
        if name == reference:
            continue
        delta_loss = []
        delta_brier = []
        for _ in range(repetitions):
            sample = np.concatenate([np.flatnonzero(bands == b) for b in rng.choice(units, len(units), replace=True)])
            delta_loss.append(log_loss(y[sample], p[name][sample], labels=[0, 1]) -
                              log_loss(y[sample], p[reference][sample], labels=[0, 1]))
            delta_brier.append(brier_score_loss(y[sample], p[name][sample]) -
                               brier_score_loss(y[sample], p[reference][sample]))
        rows.append({"model": name, "reference": reference,
                     "delta_log_loss_2.5_50_97.5": np.percentile(delta_loss, [2.5, 50, 97.5]).tolist(),
                     "delta_brier_2.5_50_97.5": np.percentile(delta_brier, [2.5, 50, 97.5]).tolist()})
    return {"repetitions": repetitions, "seed": seed, "independent_units": len(units),
            "interpretation": "Conditional block stability, fixed OOF predictions; nested fits not rerun; no p-value",
            "comparisons": rows}
