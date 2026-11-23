"""Test matching NARX input signals to dataframe columns by name."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from fastcan.narx import NARX, make_narx, print_narx

NAMES = ["u0", "u1", "u2"]
FEAT_IDS = [[-1, 0], [-1, 1], [-1, 2]]
DELAY_IDS = [[-1, 0], [-1, 0], [-1, 0]]


def _make_data(n_samples=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 3))
    y = X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2]
    return X, y


def _frame(X, names):
    return pd.DataFrame(X, columns=names)


def _narx(**kwargs):
    return NARX(feat_ids=FEAT_IDS, delay_ids=DELAY_IDS, **kwargs)


def test_fit_records_feature_names():
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    assert narx.feature_names_ is not None
    assert narx.feature_names_.tolist() == NAMES
    assert narx.feature_names_in_.tolist() == NAMES
    assert_allclose(narx.coef_, [1.0, -2.0, 0.5])


def test_predict_reordered_columns():
    """Week-2 table: the same signals in a different column order."""
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    y_hat = narx.predict(_frame(X, NAMES))

    order = [2, 0, 1]
    reordered_names = [NAMES[i] for i in order]
    y_hat_reordered = narx.predict(_frame(X[:, order], reordered_names))
    assert_allclose(y_hat_reordered, y_hat)

    # Extra columns that do not belong to the model are ignored.
    X_extra = np.c_[X[:, order], np.ones(len(X))]
    y_hat_extra = narx.predict(_frame(X_extra, reordered_names + ["time"]))
    assert_allclose(y_hat_extra, y_hat)


def test_predict_missing_column():
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    with pytest.raises(ValueError, match="Missing input columns:\n- u1"):
        narx.predict(_frame(X[:, :2], ["u0", "u2"]))
    with pytest.raises(ValueError, match="Missing input columns:"):
        narx.predict(_frame(X[:, :2], ["u0", "u1"]))


def test_predict_renamed_columns():
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    with pytest.raises(ValueError, match="Missing input columns:"):
        narx.predict(_frame(X, ["a", "b", "c"]))


def test_predict_duplicate_columns():
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    duplicate = pd.DataFrame(
        np.c_[X[:, 0], X[:, 0], X[:, 2]], columns=["u0", "u0", "u2"]
    )
    with pytest.raises(ValueError, match="duplicate column names"):
        narx.predict(duplicate)
    with pytest.raises(ValueError, match="duplicate column names"):
        _narx().fit(duplicate, y)


def test_fit_duplicate_feature_names():
    X, y = _make_data()
    with pytest.raises(ValueError, match="contains duplicate names"):
        _narx(feature_names=["u0", "u0", "u2"]).fit(_frame(X, NAMES), y)


def test_fit_feature_names_length_mismatch():
    X, y = _make_data()
    # `u3` is declared but absent from X.
    with pytest.raises(ValueError, match="Missing input columns:\n- u3"):
        _narx(feature_names=["u0", "u1", "u2", "u3"]).fit(_frame(X, NAMES), y)
    with pytest.raises(ValueError, match="without column names"):
        _narx(feature_names=["u0", "u1", "u2"]).fit(X[:, :2], y)


def test_fit_feature_names_type_error():
    X, y = _make_data()
    with pytest.raises(TypeError, match="should only contain strings"):
        _narx(feature_names=["u0", "u1", 2]).fit(_frame(X, NAMES), y)
    with pytest.raises(TypeError, match="should be a sequence of strings"):
        _narx(feature_names="u0").fit(_frame(X, NAMES), y)


def test_predict_dataframe_without_recorded_names():
    """A model fitted on arrays has no archived names: match by name is unsafe."""
    X, y = _make_data()
    narx = _narx().fit(X, y)
    assert narx.feature_names_ is None
    with pytest.raises(ValueError, match="fitted without input signal names"):
        narx.predict(_frame(X, NAMES))
    # Positional matching is still available by passing an array.
    assert_allclose(narx.predict(X), X @ np.array([1.0, -2.0, 0.5]))


def test_feature_names_parameter():
    """Declare the input signal names explicitly, independent of column order."""
    X, y = _make_data()
    narx = _narx(feature_names=NAMES).fit(X, y)
    assert narx.feature_names_.tolist() == NAMES

    order = [1, 2, 0]
    narx_reordered = _narx(feature_names=NAMES).fit(
        _frame(X[:, order], [NAMES[i] for i in order]), y
    )
    assert_allclose(narx_reordered.coef_, narx.coef_)
    assert_allclose(
        narx_reordered.predict(_frame(X[:, order], [NAMES[i] for i in order])),
        narx.predict(X),
    )

    with pytest.raises(ValueError, match="Missing input columns:\n- u2"):
        _narx(feature_names=NAMES).fit(_frame(X[:, :2], ["u0", "u1"]), y)


def test_feature_names_with_array_is_positional():
    X, y = _make_data()
    narx = _narx(feature_names=NAMES).fit(X, y)
    assert_allclose(narx.predict(X), X @ np.array([1.0, -2.0, 0.5]))
    with pytest.raises(ValueError, match="expecting 3 features"):
        narx.predict(X[:, :2])


def test_feature_names_none_inputs():
    rng = np.random.default_rng(0)
    y = rng.random(20)
    with pytest.raises(ValueError, match="should be None when X is None"):
        _narx(feature_names=NAMES).fit(None, y)


def test_non_string_columns_are_positional():
    X, y = _make_data()
    narx = _narx().fit(pd.DataFrame(X), y)
    assert narx.feature_names_ is None
    assert_allclose(narx.predict(pd.DataFrame(X)), X @ np.array([1.0, -2.0, 0.5]))


def test_unnamed_dataframe_is_rejected_when_model_has_names():
    """A dataframe without string column names cannot be matched safely."""
    X, y = _make_data()
    unnamed = pd.DataFrame(X)
    narx = _narx().fit(_frame(X, NAMES), y)
    with pytest.raises(ValueError, match="do not have string names"):
        narx.predict(unnamed)
    with pytest.raises(ValueError, match="do not have string names"):
        narx.fit(unnamed, y)

    # Same for names declared via the `feature_names` parameter.
    declared = _narx(feature_names=NAMES).fit(X, y)
    with pytest.raises(ValueError, match="do not have string names"):
        declared.predict(unnamed)

    # Passing an array keeps positional matching available.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert_allclose(narx.predict(X), X @ np.array([1.0, -2.0, 0.5]))


def test_mixed_column_name_types():
    X, y = _make_data()
    mixed = pd.DataFrame(X, columns=["u0", 1, "u2"])
    with pytest.raises(TypeError, match="all input columns have string names"):
        _narx().fit(mixed, y)


def test_pickle_round_trip():
    """The archived model must keep its input signal names after loading."""
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    loaded = pickle.loads(pickle.dumps(narx))
    assert loaded.feature_names_.tolist() == NAMES

    order = [2, 0, 1]
    loaded_names = [NAMES[i] for i in order]
    assert_allclose(
        loaded.predict(_frame(X[:, order], loaded_names)),
        narx.predict(_frame(X, NAMES)),
    )
    with pytest.raises(ValueError, match="Missing input columns:"):
        loaded.predict(_frame(X[:, :2], ["u0", "u1"]))


def test_legacy_model_feature_names_in_fallback():
    """Models archived by older versions only have `feature_names_in_`."""
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    del narx.feature_names_
    order = [2, 0, 1]
    loaded_names = [NAMES[i] for i in order]
    assert_allclose(
        narx.predict(_frame(X[:, order], loaded_names)),
        _narx().fit(_frame(X, NAMES), y).predict(_frame(X, NAMES)),
    )


def test_refit_with_reordered_columns():
    X, y = _make_data()
    narx = _narx().fit(_frame(X, NAMES), y)
    coef = np.copy(narx.coef_)

    order = [1, 2, 0]
    narx.fit(_frame(X[:, order], [NAMES[i] for i in order]), y)
    assert_allclose(narx.coef_, coef)

    with pytest.raises(ValueError, match="Missing input columns:"):
        narx.fit(_frame(X, ["a", "b", "c"]), y)


def test_make_narx_records_names():
    rng = np.random.default_rng(3)
    n_samples = 300
    u0 = rng.random(n_samples)
    u1 = rng.random(n_samples)
    u2 = rng.random(n_samples)
    y = np.zeros(n_samples)
    for i in range(3, n_samples):
        y[i] = 0.5 * y[i - 1] + 0.3 * u0[i] ** 2 + 2 * u0[i - 1] * u2[i - 3] + 1
    X = np.c_[u0, u1, u2]
    narx = make_narx(
        _frame(X, NAMES), y, n_terms_to_select=3, max_delay=3, poly_degree=2, verbose=0
    )
    assert narx.feature_names is not None
    narx.fit(_frame(X, NAMES), y)
    assert narx.feature_names_.tolist() == NAMES

    y_hat = narx.predict(_frame(X, NAMES))
    order = [2, 1, 0]
    y_hat_reordered = narx.predict(_frame(X[:, order], [NAMES[i] for i in order]))
    assert_allclose(y_hat_reordered, y_hat)

    with pytest.raises(ValueError, match="Missing input columns:"):
        narx.predict(_frame(X[:, [0, 2]], ["u0", "u2"]))


def test_predict_reordered_columns_keeps_delays():
    """Reordering the columns must not move a signal to another delay."""
    rng = np.random.default_rng(7)
    n_samples = 200
    u0 = rng.random(n_samples)
    u1 = rng.random(n_samples)
    # y depends on u0 at delay 0 and u1 at delay 2, with very different weights.
    y = np.zeros(n_samples)
    for i in range(3, n_samples):
        y[i] = 3 * u0[i] - 0.2 * u1[i - 2]
    X = np.c_[u0, u1]
    feat_ids = [[-1, 0], [-1, 1]]
    delay_ids = [[-1, 0], [-1, 2]]
    names = ["u0", "u1"]
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(_frame(X[3:], names), y[3:])
    assert_allclose(narx.coef_, [3.0, -0.2], atol=1e-6)

    y_hat = narx.predict(_frame(X[3:], names))
    # Swap the two columns in the table. Positional matching would apply the
    # delay-2 term to u0, which is a different result.
    y_hat_swapped = narx.predict(_frame(X[3:, ::-1], ["u1", "u0"]))
    assert_allclose(y_hat_swapped, y_hat)


def test_print_narx_shows_signal_names(capsys):
    """The archived model prints which signals and delays it uses."""
    X, y = _make_data()
    narx = NARX(
        feat_ids=[[-1, 0], [-1, 1], [-1, 2]],
        delay_ids=[[-1, 0], [-1, 0], [-1, 1]],
    ).fit(_frame(X, NAMES), y)
    print_narx(narx)
    out = capsys.readouterr().out
    assert "u0[k]" in out
    assert "u1[k]" in out
    assert "u2[k-1]" in out
    assert "X[k" not in out


def test_print_narx_without_names_shows_indices(capsys):
    X, y = _make_data()
    narx = NARX(feat_ids=FEAT_IDS, delay_ids=DELAY_IDS).fit(X, y)
    print_narx(narx)
    out = capsys.readouterr().out
    assert "X[k,0]" in out
    assert "X[k,1]" in out
    assert "X[k,2]" in out
