import numpy as np
import pytest

from sonnet_geometry.analysis import (cosine_similarity_matrix, fdr_adjust, linear_cka,
                                     permutation_agreement, prefix_representations, rsa,
                                     stylometric_features, trajectory_metrics, whole_sonnet_metrics)


def test_pooling_distinguishes_last_token_from_cumulative_mean_and_specials():
    h = np.array([[[100, 100], [1, 0], [0, 1], [2, 2]]], dtype=float)
    mask = [False, True, True, True]
    last = prefix_representations(h, [2, 4], pooling="last_token", include_tokens=mask)
    mean = prefix_representations(h, [2, 4], pooling="cumulative_mean", include_tokens=mask)
    np.testing.assert_allclose(last, [[[1, 0], [2, 2]]])
    np.testing.assert_allclose(mean, [[[1, 0], [1, 1]]])
    with pytest.raises(ValueError, match="increasing"):
        prefix_representations(h, [3, 2], pooling="last_token")
    with pytest.raises(ValueError, match="no included"):
        prefix_representations(h, [1], pooling="last_token", include_tokens=mask)


def test_trajectory_metrics_known_geometry_and_undefined_zero_vectors():
    result = trajectory_metrics([[[1, 0], [0, 1], [-1, 0]]])
    np.testing.assert_allclose(result["cosine_distance"], [[1, 1]])
    np.testing.assert_allclose(result["euclidean_distance"], [[np.sqrt(2), np.sqrt(2)]])
    np.testing.assert_allclose(result["turn_angle_radians"], [[np.pi / 2]])
    zero = trajectory_metrics([[[0, 0], [1, 0]]])
    assert np.isnan(zero["cosine_distance"][0, 0])


def test_anomaly_is_off_diagonal_and_invariant_to_row_scaling():
    x = np.array([[1, 0], [1, 0], [-1, 0]])
    result = whole_sonnet_metrics(x)
    np.testing.assert_allclose(result["mean_distance_anomaly"], [1, 1, 2])
    np.testing.assert_allclose(result["cosine_similarity"], cosine_similarity_matrix(x * [[3], [2], [10]]))
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity_matrix([[0, 0]])


def test_rsa_has_rank_invariance_and_tie_handling():
    rdm = np.array([[0, 1, 1, 3], [1, 0, 2, 4], [1, 2, 0, 5], [3, 4, 5, 0]])
    assert rsa(rdm, rdm ** 2) == pytest.approx(1)
    with pytest.raises(ValueError, match="constant"):
        rsa(np.zeros((3, 3)), np.zeros((3, 3)))


def test_cka_allows_unequal_dimensions_and_orthogonal_invariance():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    transformed = np.column_stack((3 * x @ q + 5, np.ones((20, 3))))
    assert linear_cka(x, transformed) == pytest.approx(1)


def test_rsa_null_permutes_whole_sonnets_and_is_reproducible():
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(9, 4)), rng.normal(size=(9, 6))
    a, b = 1 - cosine_similarity_matrix(x), 1 - cosine_similarity_matrix(y)
    result = permutation_agreement(a, b, permutations=30, seed=12)
    expected_rng = np.random.default_rng(12)
    expected = []
    for _ in range(30):
        labels = expected_rng.permutation(9)
        expected.append(rsa(a, b[np.ix_(labels, labels)]))
    np.testing.assert_allclose(result["null_values"], expected)
    assert result["p_value"] == (1 + sum(v >= result["observed"] - 1e-12 for v in expected)) / 31
    assert result["p_value"] > 0


def test_exchangeability_blocks_do_not_move_singleton_sonnets():
    x = np.random.default_rng(1).normal(size=(6, 3))
    result = permutation_agreement(x, x, statistic="linear_cka", permutations=10,
                                   blocks=list(range(6)))
    np.testing.assert_allclose(result["null_values"], 1)
    assert result["p_value"] == 1
    with pytest.raises(ValueError, match="blocks"):
        permutation_agreement(x, x, statistic="linear_cka", blocks=[1])


def test_fdr_by_handles_dependent_comparison_family_conservatively():
    p = np.array([0.01, 0.04, 0.03])
    np.testing.assert_allclose(fdr_adjust(p, method="bh"), [0.03, 0.04, 0.04])
    np.testing.assert_allclose(fdr_adjust(p), np.array([0.03, 0.04, 0.04]) * (1 + 1 / 2 + 1 / 3))
    with pytest.raises(ValueError):
        fdr_adjust([float("nan")])


def test_stylometry_is_finite_and_has_named_columns():
    matrix, names = stylometric_features(["Love, love!\nTime’s end.", "One line."])
    assert matrix.shape == (2, len(names))
    assert matrix[0, names.index("line_count")] == 2
    assert np.isfinite(matrix).all()
