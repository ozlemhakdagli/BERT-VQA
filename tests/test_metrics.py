"""Tests for official VQA accuracy metrics, answer normalization, and cluster bootstrap."""
import random

import numpy as np
import pandas as pd
import pytest

from vqa_lab.metrics import (
    ARTICLES, CONTRACTIONS, MANUAL_MAP, PUNCT, BootstrapResult,
    MissingPredictionsError, clean_whitespace, cluster_bootstrap_diff,
    cluster_bootstrap_mean, evaluate, normalize_answer, process_digit_article,
    process_punctuation, score_question,
)

EDGE_STRINGS = [
    "", " ", "   ", "?!", "a", "an", "the the", "Two dogs", "none", "zero", "ten", "dont", "Dont", "can't", "wont",
    "1,000", "1,000,000", "2.5.3", "3.5 cm.", "yes/no", "(left)", "naïve ☕", "...", "a b.", "\ttab\tseparated\n",
    "hello, world!", "The  Dog's bone", "It's 5 o'clock", "oclock", "$5.00", "50%", "red; blue", "x-ray", "snow_man",
    'he said "hi"', "[1]", "{a}", "a+b=c", "back\\slash", "<tag>", "@home", "`tick`", "1.", ".5",
]
EXP_PUNCT = [
    "", " ", "   ", "  ", "a", "an", "the the", "Two dogs", "none", "zero", "ten", "dont", "Dont", "can't", "wont",
    "1000", "1000000", "2.5.3", "3.5 cm", "yes no", " left ", "naïve ☕", "", "a b", "\ttab\tseparated\n",
    "hello world ", "The  Dog's bone", "It's 5 o'clock", "oclock", "$5.00", "50%", "red blue", "x ray", "snow man",
    "he said hi", " 1 ", " a ", "a b c", "back slash", " tag ", " home", " tick ", "1", ".5",
]
EXP_DIGIT = [
    "", "", "", "?!", "", "", "", "2 dogs", "0", "0", "10", "don't", "don't", "can't", "won't",
    "1,000", "1,000,000", "2.5.3", "3.5 cm.", "yes/no", "(left)", "naïve ☕", "...", "b.", "tab separated",
    "hello, world!", "dog's bone", "it's 5 o'clock", "o'clock", "$5.00", "50%", "red; blue", "x-ray", "snow_man",
    'he said "hi"', "[1]", "{a}", "a+b=c", "back\\slash", "<tag>", "@home", "`tick`", "1.", ".5",
]
EXP_NORM = [
    "", "", "", "", "", "", "", "2 dogs", "0", "0", "10", "don't", "don't", "can't", "won't",
    "1000", "1000000", "2.5.3", "3.5 cm", "yes no", "left", "naïve ☕", "", "b", "tab separated",
    "hello world", "dog's bone", "it's 5 o'clock", "o'clock", "$5.00", "50%", "red blue", "x ray", "snow man",
    "he said hi", "1", "", "b c", "back slash", "tag", "home", "tick", "1", ".5",
]

EDGE_CASES = [
    ("yes", ["yes"] * 10, 1.0), ("no", ["yes"] * 10, 0.0), ("yes", ["Yes"] * 10, 0.0), ("Yes", ["Yes"] * 10, 1.0),
    ("cat", ["cat"] * 3 + ["dog"] * 7, 0.9), ("cat", ["cat"] + ["dog"] * 9, 0.3), ("cat", ["cat"] * 2 + ["dog"] * 8, 0.6),
    ("", ["cat"] * 10, 0.0), ("", [""] * 10, 1.0), ("dog\n", ["dog"] * 5 + ["cat"] * 5, 1.0), ("  dog  ", ["dog"] * 5 + ["cat"] * 5, 1.0),
    ("dog", ["dog\n"] * 3 + ["cat"] * 7, 0.9), ("Café", ["café"] * 4 + ["cafe"] * 6, 1.0), ("?", ["?"] * 3 + ["yes"] * 7, 0.9),
    ("two", ["2"] * 5 + ["3"] * 5, 1.0), ("2", ["two"] * 5 + ["3"] * 5, 1.0), ("the dog", ["dog"] * 5 + ["cat"] * 5, 1.0),
    ("A cell phone.", ["cell phone"] * 6 + ["phone"] * 4, 1.0), ("dont", ["don't"] * 5 + ["no"] * 5, 1.0),
    ("don't", ["dont"] * 5 + ["no"] * 5, 1.0), ("1,000", ["1000"] * 5 + ["999"] * 5, 1.0),
    ("red and white", ["red and white"] * 2 + ["red"] * 8, 0.6), ("RED", ["red"] * 3 + ["blue"] * 7, 0.9),
    ("yes", ["yes"] * 9 + ["no"], 1.0), ("no", ["yes"] * 9 + ["no"], 0.3),
    ("man", ["man", "woman", "man", "boy", "girl", "man", "men", "person", "people", "man"], 1.0),
]

BASE_ANSWERS = [
    "yes", "no", "2", "3", "red", "blue", "tennis", "cat", "dog", "left", "right", "man", "woman", "kitchen", "frisbee",
    "pizza", "white", "black", "0", "1", "banana", "snow", "wood", "sunny", "cloudy", "umbrella", "bus", "train", "grass", "water",
]
VARIANTS = [lambda s: s, lambda s: s.capitalize(), lambda s: s + ".", lambda s: "the " + s, lambda s: " " + s + " ",
            lambda s: {"2": "two", "3": "three", "0": "none", "1": "one"}.get(s, s)]


def random_corpus(n=400, seed=0):
    """Generate a synthetic ground-truth question DataFrame and matching predictions.

    Args:
        n: Number of questions to generate. Defaults to 400.
        seed: Random seed for generator. Defaults to 0.

    Returns:
        tuple[pd.DataFrame, dict[int, str]]: Synthetic ground truth and predictions dictionary.
    """
    rng = random.Random(seed)
    rows, preds = [], {}
    for qid in range(1, n + 1):
        main, other = rng.sample(BASE_ANSWERS, 2)
        k = rng.randint(1, 10)
        answers = [rng.choice(VARIANTS)(main) for _ in range(k)] + [rng.choice(VARIANTS)(other) for _ in range(10 - k)]
        rows.append({"question_id": qid, "question_type": rng.choice(["what", "is the", "how many", "are there"]),
                     "answer_type": rng.choice(["yes/no", "number", "other"]), "answers_raw": answers})
        preds[qid] = rng.choice(VARIANTS)(rng.choice([main, other, rng.choice(BASE_ANSWERS)]))
    return pd.DataFrame(rows), preds


def test_normalisation():
    """Test answer normalization pipeline on edge cases and upstream quirks."""
    assert len(CONTRACTIONS) == 120 and CONTRACTIONS["dont"] == "don't" and CONTRACTIONS["somebody'd"] == "somebodyd"
    assert len(MANUAL_MAP) == 12 and MANUAL_MAP["none"] == "0" and MANUAL_MAP["ten"] == "10"
    assert ARTICLES == ("a", "an", "the") and len(PUNCT) == 21
    assert [process_punctuation(s) for s in EDGE_STRINGS] == EXP_PUNCT
    assert [process_digit_article(s) for s in EDGE_STRINGS] == EXP_DIGIT
    assert [clean_whitespace(s) for s in EDGE_STRINGS] == [s.replace("\n", " ").replace("\t", " ").strip() for s in EDGE_STRINGS]
    assert [normalize_answer(s) for s in EDGE_STRINGS] == EXP_NORM
    assert process_punctuation(".".join(["a"] * 40)) == "a" * 33 + ".a" * 7   # re.UNICODE (32) passed as sub count


def test_score_question_and_evaluate():
    """Verify leave-one-out soft accuracy and DataFrame evaluation against official behavior."""
    for pred, answers, exp in EDGE_CASES:
        assert score_question(pred, answers) == pytest.approx(exp)
    # the official script sums one value per human answer in answer order, so the last bit depends on where the matches are
    assert score_question("cat", ["cat" if i in (0, 1, 9) else "dog" for i in range(10)]) == 0.8999999999999998
    assert score_question("cat", ["cat"] * 3 + ["dog"] * 7) == 0.9
    gt, preds = random_corpus()
    result = evaluate(preds, gt)
    assert list(result.per_question.index) == gt["question_id"].tolist()
    assert result.per_question.tolist() == [score_question(preds[r.question_id], r.answers_raw) for r in gt.itertuples()]
    assert result.overall == pytest.approx(100 * float(result.per_question.mean()))


def test_missing_and_extra_predictions():
    """Test handling of missing predictions (raise vs zero) and extraneous predictions."""
    gt, preds = random_corpus(n=10)
    baseline = evaluate(preds, gt)
    assert evaluate({**preds, 999: "yes", 1000: "no"}, gt).overall == baseline.overall
    for q in (2, 5, 9):
        del preds[q]
    with pytest.raises(MissingPredictionsError) as exc:
        evaluate(preds, gt)
    assert "3" in str(exc.value) and "2" in str(exc.value)
    zeroed = evaluate(preds, gt, missing="zero")
    assert len(zeroed.per_question) == 10 and zeroed.per_question[2] == zeroed.per_question[5] == zeroed.per_question[9] == 0.0


def test_cluster_bootstrap_mean():
    """Test cluster bootstrap confidence interval estimation over image clusters."""
    rng = np.random.default_rng(42)
    clusters = rng.integers(0, 200, size=1000)
    scores = rng.normal(0.5, 0.2, size=200)[clusters] + rng.normal(0.0, 0.05, size=1000)
    r1 = cluster_bootstrap_mean(scores, clusters, n_boot=300, seed=42)
    r2 = cluster_bootstrap_mean(scores, clusters, n_boot=300, seed=42)
    r3 = cluster_bootstrap_mean(scores, clusters, n_boot=300, seed=43)
    assert r1.estimate == scores.mean() and r1.lo < r1.estimate < r1.hi
    assert r1 == r2 and (r1.lo != r3.lo or r1.hi != r3.hi)

    c_means = rng.normal(0.5, 0.3, size=200)
    c_scores, c_clust = np.repeat(c_means, 5), np.repeat(np.arange(200), 5)
    w_clust = cluster_bootstrap_mean(c_scores, c_clust, n_boot=500, seed=0)
    w_unclust = cluster_bootstrap_mean(c_scores, np.arange(1000), n_boot=500, seed=0)
    assert (w_clust.hi - w_clust.lo) > (w_unclust.hi - w_unclust.lo)
    same = cluster_bootstrap_mean(np.full(30, 0.7), np.arange(30) % 3)   # identical scores: a zero-width interval, not NaN
    assert same.hi - same.lo < 1e-12 and same.lo == pytest.approx(0.7) and same.estimate == pytest.approx(0.7)
    two = cluster_bootstrap_mean([0.0, 1.0, 1.0], [0, 1, 1])             # two clusters still give a finite interval
    assert 0.0 <= two.lo <= two.hi <= 1.0
    one = BootstrapResult(0.5, 0.5, 0.5)                                 # one cluster: nothing to resample, the estimate itself
    assert cluster_bootstrap_mean([0.0, 1.0], [7, 7]) == one and cluster_bootstrap_diff([1.0], [0.5], [3]) == one


def test_cluster_bootstrap_diff():
    """Test paired difference cluster bootstrap and validation of input array shapes."""
    rng = np.random.default_rng(42)
    s, c = rng.random(100), rng.integers(0, 20, size=100)
    res_zero = cluster_bootstrap_diff(s, s, c, n_boot=200, seed=0)
    assert res_zero.estimate == 0.0 and res_zero.lo == 0.0 and res_zero.hi == 0.0

    rng = np.random.default_rng(0)
    a = rng.random(2000)
    res_shift = cluster_bootstrap_diff(a, a + 0.1, rng.integers(0, 400, size=2000), n_boot=500, seed=0)
    assert res_shift.estimate == pytest.approx(-0.1) and res_shift.hi < 0.0 and (res_shift.hi - res_shift.lo) < 1e-9

    one, two = np.array([1.0]), np.array([1.0, 2.0])
    for call in (
        lambda: cluster_bootstrap_mean(np.array([]), np.array([])),
        lambda: cluster_bootstrap_mean(two, np.array([1])),
        lambda: cluster_bootstrap_diff(two, one, np.array([1, 2])),
        lambda: cluster_bootstrap_diff(two, two, np.array([1])),
        lambda: cluster_bootstrap_mean(np.ones((2, 2)), np.ones((2, 2))),
    ):
        with pytest.raises(ValueError):
            call()
