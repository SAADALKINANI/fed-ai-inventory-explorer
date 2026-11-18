"""Text models over use-case descriptions.

* :class:`TopicClassifier` learns the 2025 ``topic_area`` from the free text
  and fills in the rows where agencies left it blank.
* :func:`discover_themes` finds recurring themes with NMF topic modelling.
* :class:`UseCaseSearch` finds existing federal use cases similar to an idea.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

EXTRA_STOP = {"ai", "use", "used", "using", "data", "model", "models", "tool", "tools", "agency", "provide",
              "provides", "help", "helps", "allow", "allows", "process", "new", "based", "information", "will",
              "system", "systems", "case", "able", "make", "also"}
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)


def text_of(df: pd.DataFrame) -> pd.Series:
    return (df["use_case_name"].fillna("") + ". " + df["purpose"].fillna("") + " " + df["outputs"].fillna("")).str.strip()


class TopicClassifier:
    def __init__(self, C: float = 4.0, seed: int = 0):
        self.seed = seed
        self.pipe = make_pipeline(
            TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    def evaluate(self, df: pd.DataFrame, label: str = "topic_area", test_size: float = 0.2) -> dict:
        d = df[df[label].notna()]
        X_tr, X_te, y_tr, y_te = train_test_split(text_of(d), d[label], test_size=test_size,
                                                  random_state=self.seed, stratify=d[label])
        self.pipe.fit(X_tr, y_tr)
        pred = self.pipe.predict(X_te)
        rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
        return {"accuracy": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"],
                "weighted_f1": rep["weighted avg"]["f1-score"], "n_test": len(y_te), "report": rep}

    def fit(self, df: pd.DataFrame, label: str = "topic_area") -> "TopicClassifier":
        d = df[df[label].notna()]
        self.pipe.fit(text_of(d), d[label])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipe.predict_proba(text_of(df))
        classes = self.pipe.classes_
        best = proba.argmax(1)
        return pd.DataFrame({"predicted_topic": classes[best], "confidence": proba.max(1).round(3)}, index=df.index)

    def fill_missing(self, df: pd.DataFrame, label: str = "topic_area", min_confidence: float = 0.5) -> pd.DataFrame:
        out = df.copy()
        miss = out[label].isna() & text_of(out).str.len().gt(20)
        pred = self.predict(out[miss])
        keep = pred.confidence >= min_confidence
        out[f"{label}_filled"] = out[label]
        out.loc[pred.index[keep], f"{label}_filled"] = pred.loc[keep, "predicted_topic"]
        out[f"{label}_imputed"] = False
        out.loc[pred.index[keep], f"{label}_imputed"] = True
        return out


def discover_themes(df: pd.DataFrame, n_themes: int = 12, n_words: int = 8, seed: int = 0) -> pd.DataFrame:
    """NMF themes: top words and how many use cases load mainly on each theme."""
    texts = text_of(df)
    vec = TfidfVectorizer(stop_words=STOP, min_df=5, max_df=0.4, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_themes, random_state=seed, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    dominant = W.argmax(1)
    rows = []
    for k, comp in enumerate(nmf.components_):
        idx = dominant == k
        example = df.loc[idx, "use_case_name"].iloc[np.argmax(W[idx, k])] if idx.any() else ""
        rows.append({"theme": k, "top_terms": ", ".join(vocab[np.argsort(-comp)[:n_words]]),
                     "use_cases": int(idx.sum()), "example": example})
    return pd.DataFrame(rows).sort_values("use_cases", ascending=False).reset_index(drop=True)


class UseCaseSearch:
    """Find federal use cases similar to a free-text idea (TF-IDF cosine)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.X = self.vec.fit_transform(text_of(self.df))

    def query(self, text: str, top_k: int = 5) -> pd.DataFrame:
        sims = (self.X @ self.vec.transform([text]).T).toarray().ravel()
        idx = np.argsort(-sims)[:top_k]
        out = self.df.loc[idx, ["year", "agency", "use_case_name", "stage", "topic_area"]].copy()
        out["similarity"] = sims[idx].round(3)
        return out.reset_index(drop=True)
