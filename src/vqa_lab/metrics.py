# Licence of the VQA evaluation tools that the answer normalisation and the accuracy below are ported from
# (https://github.com/GT-Vision-Lab/VQA, Python 2); the notice has to travel with the ported code:
#
# Copyright (c) 2014, Aishwarya Agrawal
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
#
# The views and conclusions contained in the software and documentation are those of the authors and should not
# be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.
"""Official VQA accuracy, bug-compatible with the official evaluation script, and the cluster bootstrap over images.

Includes answer text normalization (whitespace, punctuation, digits, articles, contractions).
Adapted from the VQA evaluation tools by Aishwarya Agrawal (https://github.com/GT-Vision-Lab/VQA, BSD licence).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

# =============================================================================
# Answer normalization (Aishwarya Agrawal, GT-Vision-Lab/VQA)
# =============================================================================

CONTRACTIONS: dict[str, str] = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}

MANUAL_MAP: dict[str, str] = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

ARTICLES: tuple[str, ...] = ('a', 'an', 'the')

PUNCT: tuple[str, ...] = (
    ';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!'
)

_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")


def clean_whitespace(text: str) -> str:
    """Clean and standardize whitespace characters in answer text.

    Replaces newline and tab characters with spaces and strips leading/trailing
    whitespace. Applied unconditionally to every answer.

    Args:
        text: Input answer string.

    Returns:
        str: Cleaned text with normalized whitespace.
    """
    return text.replace("\n", " ").replace("\t", " ").strip()


@lru_cache(maxsize=262_144)
def process_punctuation(text: str) -> str:
    """Process and normalize punctuation characters in accordance with official VQA tools.

    Deletes or replaces punctuation symbols with spaces, then strips every period that is not
    followed by a digit (the official script's rule, including its limit of 32 substitutions).

    Args:
        text: Input text string.

    Returns:
        str: Text with normalized punctuation.
    """
    out = text
    has_comma = _COMMA_STRIP.search(text) is not None
    for p in PUNCT:
        if p in out:
            out = out.replace(p, "" if (p + " " in text or " " + p in text or has_comma) else " ")
    # the official code passes re.UNICODE (== 32) as the *count* argument; replicated on purpose
    return _PERIOD_STRIP.sub("", out, count=int(re.UNICODE))


@lru_cache(maxsize=262_144)
def process_digit_article(text: str) -> str:
    """Normalize digits, articles, and contractions according to official VQA rules.

    Converts text to lowercase, maps number words to digits, removes articles ('a', 'an', 'the'),
    and expands known contractions.

    Args:
        text: Input text string.

    Returns:
        str: Normalized text string without articles and with expanded contractions.
    """
    return " ".join(CONTRACTIONS.get(m, m) for w in text.lower().split() if (m := MANUAL_MAP.get(w, w)) not in ARTICLES)


def normalize_answer(text: str) -> str:
    """Apply standard unconditional normalization pipeline to an answer string.

    Applies whitespace cleaning, punctuation processing, digit/article mapping,
    and contraction expansion sequentially.

    Args:
        text: Raw answer string.

    Returns:
        str: Fully normalized answer string.
    """
    return process_digit_article(process_punctuation(clean_whitespace(text)))


# =============================================================================
# Accuracy metrics and scoring
# =============================================================================

class MissingPredictionsError(KeyError):
    """Raised when ground-truth questions are missing from the prediction mapping."""


@dataclass(frozen=True)
class EvalResult:
    """Container holding evaluation accuracy metrics and breakdowns."""
    overall: float  # 0..100, unrounded
    per_question: pd.Series  # index question_id, values 0..1
    per_answer_type: dict[str, float]  # 0..100, unrounded


def soft_score(matches: int, n: int) -> float:
    """Compute the official leave-one-out soft VQA accuracy score.

    Calculates the mean accuracy of an answer agreed upon by `matches` out of `n`
    human annotators, computed over all `n` subsets of size n - 1.

    Args:
        matches: Number of human annotators who gave this exact answer.
        n: Total number of human annotations for the question (typically 10).

    Returns:
        float: Soft accuracy score between 0.0 and 1.0.
    """
    return (matches * min(1.0, (matches - 1) / 3) + (n - matches) * min(1.0, matches / 3)) / n


def score_question(prediction: str, gt_answers: Sequence[str]) -> float:
    """Compute the leave-one-out soft accuracy of a single prediction against human answers.

    Normalization is conditionally applied to the prediction and ground-truth answers
    only if the ground-truth answers are not already all identical, exactly replicating
    the official evaluation tool's behavior.

    Args:
        prediction: Model prediction string.
        gt_answers: Sequence of human ground-truth answer strings (typically 10).

    Returns:
        float: Accuracy score in the range [0.0, 1.0].
    """
    pred = clean_whitespace(prediction)
    gts = [clean_whitespace(a) for a in gt_answers]
    if len(set(gts)) > 1:   # (normalize_answer cleans the whitespace once more: a no-op on cleaned text)
        gts = [normalize_answer(a) for a in gts]
        pred = normalize_answer(pred)
    n_match = gts.count(pred)
    hit, miss = min(1, (n_match - 1) / 3), min(1, n_match / 3)   # the value of an answer equal to the prediction, and of one that is not
    return sum(hit if a == pred else miss for a in gts) / len(gts)   # summed in answer order like the official loop: the same bits


def evaluate(
    predictions: Mapping[int, str],
    gt: pd.DataFrame,
    *,
    missing: Literal["raise", "zero"] = "raise",
) -> EvalResult:
    """Score model predictions against ground truth questions and annotations.

    Questions are processed in `gt` row order. Predictions for unknown questions
    are ignored. Questions without predictions either raise an error or score zero
    depending on `missing`.

    Args:
        predictions: Mapping from question_id to predicted answer string.
        gt: Ground-truth DataFrame containing question_id, answer_type, and answers_raw.
        missing: Strategy for handling missing predictions; either 'raise' to raise
            MissingPredictionsError or 'zero' to assign score 0.0.

    Returns:
        EvalResult: Object containing overall accuracy, per-question score Series,
            and a breakdown by answer type.

    Raises:
        MissingPredictionsError: If `missing='raise'` and any ground truth questions
            lack predictions.
    """
    qids = gt["question_id"].tolist()
    absent = [q for q in qids if q not in predictions]
    if absent and missing == "raise":
        raise MissingPredictionsError(
            f"{len(absent)} of {len(qids)} questions have no prediction; first missing question_ids: {absent[:5]}"
        )
    scores = pd.Series([score_question(predictions[q], a) if q in predictions else 0.0 for q, a in zip(qids, gt["answers_raw"])],
                       index=pd.Index(qids, name="question_id"), dtype=float)

    per_answer_type = (100 * scores.groupby(gt["answer_type"].to_numpy(), sort=False).mean()).to_dict()   # first-appearance order
    return EvalResult(overall=100 * float(scores.mean()), per_question=scores, per_answer_type=per_answer_type)


class BootstrapResult(NamedTuple):
    """Results of a cluster bootstrap confidence interval calculation."""
    estimate: float
    lo: float
    hi: float


def cluster_bootstrap_mean(scores, clusters, *, n_boot: int = 1000, seed: int = 0, ci: float = 0.95) -> BootstrapResult:
    """Compute cluster-bootstrap confidence interval for the mean of scores clustered by image.

    Resamples clusters (images) with replacement using `scipy.stats.bootstrap`
    to obtain a percentile bootstrap interval accounting for question correlation
    within the same image.

    Args:
        scores: 1-D array of question scores.
        clusters: 1-D array of cluster identifiers (image IDs) matching `scores`.
        n_boot: Number of bootstrap resamples. Defaults to 1000.
        seed: Random seed for reproducible bootstrap sampling. Defaults to 0.
        ci: Confidence level for the interval. Defaults to 0.95.

    Returns:
        BootstrapResult: Named tuple with estimate, lower bound, and upper bound.

    Raises:
        ValueError: If `scores` and `clusters` are empty or have unequal lengths.
    """
    scores, clusters = np.asarray(scores, dtype=float), np.asarray(clusters)
    if scores.ndim != 1 or scores.shape != clusters.shape or scores.size == 0:
        raise ValueError("scores and clusters must be non-empty 1-D arrays of the same length")
    _, group = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(group, weights=scores), np.bincount(group)
    est = float(scores.mean())
    if len(sums) < 2 or not scores.any():   # one cluster (scipy needs two) or all zeros: every resample is the sample itself
        return BootstrapResult(est, est, est)
    res = bootstrap((sums, counts), lambda s, c, axis=-1: s.sum(axis=axis) / c.sum(axis=axis), paired=True, vectorized=True,
                    n_resamples=n_boot, confidence_level=ci, method="percentile",
                    batch=100,   # batches keep memory small for ~40,000 clusters
                    rng=np.random.default_rng(seed))
    return BootstrapResult(est, float(res.confidence_interval.low), float(res.confidence_interval.high))


def cluster_bootstrap_diff(scores_a, scores_b, clusters, **kwargs) -> BootstrapResult:
    """Compute paired difference between two models using cluster bootstrap over images.

    Both systems are evaluated on the exact same resampled clusters to compute
    the bootstrap distribution of differences `scores_a - scores_b`.

    Args:
        scores_a: 1-D array of scores for the first model.
        scores_b: 1-D array of scores for the second model.
        clusters: 1-D array of cluster identifiers (image IDs).
        **kwargs: Additional keyword arguments passed to `cluster_bootstrap_mean`.

    Returns:
        BootstrapResult: Named tuple with estimate, lower bound, and upper bound of the difference.

    Raises:
        ValueError: If `scores_a` and `scores_b` have different lengths.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:  # otherwise numpy would silently broadcast
        raise ValueError("scores_a and scores_b must have the same length")
    return cluster_bootstrap_mean(a - b, clusters, **kwargs)
