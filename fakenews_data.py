"""Shared data code for every notebook: loading, the fixed split, and the shortcut stress tests."""

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEED = 42
DATASET_REVISION = "9d489de0df0f7215914660b00d478a4e2069e850"
DATASET_URL = f"https://huggingface.co/datasets/jcblaise/fake_news_filipino/resolve/{DATASET_REVISION}/fakenews.zip"
LABEL_NAMES = {0: "Real", 1: "Fake"}
TEST_FINGERPRINT = "1c312f3fbfe62465"

PLATFORM_WORDS = [
    "source", "sources", "video", "videos", "facebook", "fb", "youtube", "twitter",
    "panoorin", "news", "photo", "photos", "credits", "share", "via", "post",
]


def load_articles(data_dir):
    data_dir = Path(data_dir)
    csv_path = data_dir / "fakenews" / "full.csv"
    if not csv_path.exists():
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(data_dir)

    raw = pd.read_csv(csv_path)
    raw["article"] = raw["article"].str.strip()
    raw["label"] = raw["label"].astype(int)
    return raw


def drop_duplicates(raw):
    return raw.drop_duplicates(subset="article", keep="first").reset_index(drop=True)


def make_splits(articles):
    train_df, holdout_df = train_test_split(
        articles, test_size=0.30, stratify=articles["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        holdout_df, test_size=0.50, stratify=holdout_df["label"], random_state=SEED
    )
    return train_df, val_df, test_df


def split_fingerprint(df):
    joined = "\n".join(sorted(df["article"]))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def load_splits(data_dir):
    articles = drop_duplicates(load_articles(data_dir))
    train_df, val_df, test_df = make_splits(articles)
    if split_fingerprint(test_df) != TEST_FINGERPRINT:
        raise ValueError("Test split does not match notebook 01. Check the dataset revision and seed.")
    return train_df, val_df, test_df


def word_count(texts):
    return texts.str.split().str.len()


def length_matched(df, n_bins=10):
    """Keep equal numbers of real and fake articles inside each length bin.

    After this, article length alone carries no information about the label.
    """
    log_length = np.log1p(word_count(df["article"]))
    bins = pd.qcut(log_length, q=n_bins, labels=False, duplicates="drop")
    rng = np.random.RandomState(SEED)

    kept = []
    for bin_id in sorted(bins.unique()):
        in_bin = df[bins == bin_id]
        real = in_bin[in_bin["label"] == 0]
        fake = in_bin[in_bin["label"] == 1]
        n_keep = min(len(real), len(fake))
        kept.append(real.sample(n=n_keep, random_state=rng))
        kept.append(fake.sample(n=n_keep, random_state=rng))
    return pd.concat(kept).sort_index()


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NUMBER_PATTERN = re.compile(r"\d+([.,]\d+)*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PLATFORM_PATTERN = re.compile(r"\b(" + "|".join(PLATFORM_WORDS) + r")\b", flags=re.IGNORECASE)


def mask_shortcuts(text):
    """Remove the cues that identify the source or time period rather than the content.

    Numbers (including years) become xxnum. A capitalized word that does not start a sentence is
    treated as a name and becomes xxname. URLs and platform words are dropped. Everything is
    lowercased, which also hides the all-caps first word that tabloid articles use.
    """
    text = URL_PATTERN.sub(" ", text)
    text = PLATFORM_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" xxnum ", text)

    masked_sentences = []
    for sentence in SENTENCE_SPLIT.split(text):
        tokens = sentence.split()
        for i in range(1, len(tokens)):
            if tokens[i][0].isupper():
                tokens[i] = "xxname"
        masked_sentences.append(" ".join(tokens))

    return " ".join(masked_sentences).lower()


def masked_copy(df):
    out = df.copy()
    out["article"] = out["article"].map(mask_shortcuts)
    return out


def build_test_sets(test_df):
    test_sets = {
        "standard": test_df,
        "length_matched": length_matched(test_df),
    }
    test_sets["masked"] = masked_copy(test_sets["standard"])
    test_sets["masked_length_matched"] = masked_copy(test_sets["length_matched"])
    return test_sets


def bootstrap_macro_f1(y_true, y_pred, n_resamples=1000):
    """95% interval for macro F1 from resampling the test articles with replacement."""
    rng = np.random.RandomState(SEED)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return np.percentile(scores, [2.5, 97.5])
