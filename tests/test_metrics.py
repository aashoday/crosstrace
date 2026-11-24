import pytest

from crosstrace.metrics import evaluate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_single_gold_in_third_place():
    ranked = ["a", "b", "c", "d"]
    gold = {"c"}
    assert reciprocal_rank(ranked, gold) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, gold, 2) == 0.0
    assert recall_at_k(ranked, gold, 3) == 1.0
    # DCG = 1 / log2(4) = 0.5. IDCG = 1 / log2(2) = 1.
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(0.5)


def test_two_gold_items():
    ranked = ["a", "x", "b", "y"]
    gold = {"a", "b"}
    assert recall_at_k(ranked, gold, 1) == 0.5
    assert recall_at_k(ranked, gold, 3) == 1.0
    assert reciprocal_rank(ranked, gold) == 1.0
    # DCG = 1 + 1/log2(4) = 1.5. IDCG = 1 + 1/log2(3) = 1.63093.
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(0.9197, abs=1e-4)


def test_perfect_ranking_has_ndcg_one():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 5) == pytest.approx(1.0)


def test_gold_not_in_list_scores_zero():
    ranked = ["a", "b"]
    gold = {"z"}
    assert recall_at_k(ranked, gold, 5) == 0.0
    assert reciprocal_rank(ranked, gold) == 0.0
    assert ndcg_at_k(ranked, gold, 5) == 0.0


def test_empty_ranked_list_scores_zero():
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a"}, 5) == 0.0


def test_duplicates_do_not_change_the_rank():
    # After removing the duplicate, "b" is at rank 2.
    assert reciprocal_rank(["a", "a", "b"], {"b"}) == pytest.approx(0.5)


def test_k_larger_than_list_is_fine():
    assert recall_at_k(["a"], {"a"}, 100) == 1.0


def test_empty_gold_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], set())
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 5)


def test_k_below_one_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, 0)


def test_evaluate_means_over_all_queries():
    run = {"q1": ["a", "b"], "q2": [], "extra": ["x"]}
    qrels = {"q1": {"b"}, "q2": {"z"}, "q3": {"c"}}
    result = evaluate(run, qrels, ks=(1, 2))
    # q3 is missing from run, so it scores 0. "extra" is ignored.
    assert result["n_queries"] == 3
    assert result["recall@1"] == 0.0
    assert result["recall@2"] == pytest.approx(1 / 3)
    assert result["mrr"] == pytest.approx(0.5 / 3)


def test_evaluate_needs_queries():
    with pytest.raises(ValueError):
        evaluate({}, {})