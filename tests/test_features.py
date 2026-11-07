import numpy as np

from cage.features import COMPARISON_FEATURE_NAMES, pairwise_comparison_features


def test_comparison_feature_schema_is_stable():
    features = pairwise_comparison_features(
        "int add(int a) { return a + 1; }",
        "int sum(int b) { return b + 1; }",
        tfidf_cosine=0.5,
    )
    assert features.shape == (20,)
    assert features.dtype == np.float64
    assert np.isfinite(features).all()


def test_comparison_feature_order_matches_experiment_contract():
    a, b = "int add(int x) { return x + 1; }", "int sum(int x) { int y = x; return y; }"
    values = pairwise_comparison_features(a, b, tfidf_cosine=0.4)
    assert COMPARISON_FEATURE_NAMES[13:19] == (
        "sim_mean", "sim_std", "sim_max", "sim_min", "token_ratio", "token_diff"
    )
    np.testing.assert_allclose(values[13:17], [
        values[:6].mean(), values[:6].std(ddof=1), values[:6].max(), values[:6].min()
    ])
    assert np.isclose(values[17], len(a.split()) / (len(b.split()) + 1e-6))
    assert values[18] == abs(len(a.split()) - len(b.split()))


def test_empty_inputs_are_finite():
    features = pairwise_comparison_features("", "", tfidf_cosine=0.0)
    assert features.shape == (20,)
    assert np.isfinite(features).all()
