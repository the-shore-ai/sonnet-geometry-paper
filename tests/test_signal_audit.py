import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from sonnet_geometry.signal_audit import (
    cka_from_grams, cloud_summary, neighbors, paired_metrics, partial_rank,
    rank_correlation, rdm, retrieval, seed_repeatability, unbiased_hsic,
)


def test_identity_and_constant_ties():
    x = np.eye(8)
    assert retrieval(x, x)["top1"] == 1
    constant = np.ones((8, 3))
    assert retrieval(constant, constant)["top1"] == pytest.approx(1 / 8)
    assert rank_correlation(np.zeros(5), np.arange(5)) is None
    assert cloud_summary(constant)["effective_dimension"] is None


def test_retrieval_permutation_and_bidirectionality():
    x = np.eye(8)
    y = np.roll(x, 1, axis=0)
    assert retrieval(x, y)["top1"] == 0
    assert retrieval(x, y)["mrr"] == retrieval(y, x)["mrr"]


def test_cosine_and_effective_rank_against_scipy_and_svd():
    x = np.random.default_rng(42).normal(size=(13, 7))
    np.testing.assert_allclose(rdm(x), squareform(pdist(x, metric="cosine")), atol=1e-14)
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    singular = np.linalg.svd(u - u.mean(0), compute_uv=False)
    eigenvalues = singular ** 2
    assert cloud_summary(x)["effective_dimension"] == pytest.approx(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))


def test_unbiased_hsic_against_explicit_u_centering():
    rng = np.random.default_rng(13)
    a, b = rng.normal(size=(12, 7)), rng.normal(size=(12, 9))
    k, l = a @ a.T, b @ b.T
    def ucenter(g):
        g = g.copy()
        n = len(g)
        np.fill_diagonal(g, 0)
        sums = g.sum(0)
        h = g - sums[None] / (n - 2) - sums[:, None] / (n - 2) + g.sum() / ((n - 1) * (n - 2))
        np.fill_diagonal(h, 0)
        return h
    expected = np.sum(ucenter(k) * ucenter(l)) / (len(k) * (len(k) - 3))
    assert unbiased_hsic(k, l) == pytest.approx(expected, abs=1e-13)


def test_cka_scale_rotation_and_high_dimensional_bias():
    rng = np.random.default_rng(51)
    a, b = rng.normal(size=(24, 10000)), rng.normal(size=(24, 10000))
    result = cka_from_grams(a @ a.T, b @ b.T)
    assert result["unit_gram_biased_cka"] > .98
    assert abs(result["unit_gram_unbiased_cka"]) < .2
    same = cka_from_grams(a @ a.T, 3 * (a @ a.T))
    assert same["unit_gram_biased_cka"] == pytest.approx(1)
    assert same["unit_gram_unbiased_cka"] == pytest.approx(1)


def test_neighbors_ties_and_chance_adjustment():
    d = rdm(np.random.default_rng(3).normal(size=(24, 12)))
    result = neighbors(d, d)
    assert result["mean_chance_adjusted_overlap"] == pytest.approx(1)
    constant = np.ones((24, 24)) - np.eye(24)
    assert neighbors(constant, constant)["mean_chance_adjusted_overlap"] is None
    assert neighbors(constant, constant)["mean_neighbor_set_sizes"] == [23., 23.]


def test_partial_rank_keeps_perfectly_explained_outcomes_undefined():
    x = np.arange(20, dtype=float)
    assert partial_rank(x, x ** 2, [x]) is None
    y = np.random.default_rng(4).normal(size=20)
    assert partial_rank(y, y, [x]) == pytest.approx(1)


def test_repeatability_and_paired_metrics_do_not_confuse_changes_with_noise():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(24, 30)), rng.normal(size=(24, 30))
    result = seed_repeatability(a, [b] * 5)
    np.testing.assert_allclose(result["interseed_rdm_rhos"], 1)
    assert result["between_fraction_of_total_shuffle_variance"] == pytest.approx(1)
    assert result["within_poem_seed_variance"] < 1e-25
    assert result["centroid_rdm_rho_to_original"] < .3
    same = paired_metrics(a, a)
    assert same["identity"]["top1"] == 1
    assert same["rdm_rho"] == pytest.approx(1)
    assert same["median_cosine_displacement"] < 1e-14


def test_degenerate_inputs_fail_without_fabricated_values():
    with pytest.raises(ValueError):
        rdm(np.zeros((4, 8)))
    with pytest.raises(ValueError):
        rank_correlation([1, 2], [1, np.nan])
