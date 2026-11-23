"""
Helpers to match the input signals of a NARX model to dataframe columns by name.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.exceptions import DuplicateError
from sklearn.utils import column_or_1d

_MAX_NAMES_IN_MSG = 20


def _format_names(names):
    """Format names for an error message."""
    names = list(names)
    lines = [f"- {name}" for name in names[:_MAX_NAMES_IN_MSG]]
    if len(names) > _MAX_NAMES_IN_MSG:
        lines.append(f"- ... ({len(names) - _MAX_NAMES_IN_MSG} more)")
    return "\n".join(lines)


def _get_column_names(X):
    """Return the string column names of ``X``, or None.

    Parameters
    ----------
    X : array-like
        The input data.

    Returns
    -------
    names : ndarray of shape (n_columns,), dtype=object or None
        The column names when ``X`` is a dataframe whose columns are all
        strings. None for array-like inputs or dataframes that do not have
        string column names.
    """
    if not nw.dependencies.is_into_dataframe(X):
        return None
    try:
        frame = nw.from_native(X)
    except DuplicateError as exc:
        raise ValueError(
            "X has duplicate column names. Each input signal must have a unique "
            "column name so that the NARX model can match the columns of X to "
            "the input signals of the model.\n"
            f"{exc}"
        ) from exc
    columns = list(frame.columns)
    if len(columns) == 0:
        return None
    is_str = [isinstance(name, str) for name in columns]
    if all(is_str):
        return np.asarray(columns, dtype=object)
    if any(is_str):
        types = sorted({type(name).__name__ for name in columns})
        raise TypeError(
            "Feature names are only supported if all input columns have string "
            f"names, but X has the following column name types: {types}. "
            "Convert the column names to strings, e.g., "
            "`X.columns = X.columns.astype(str)`, or remove the column names."
        )
    # Only non-string column names, e.g., integer column labels. They cannot be
    # used to identify input signals. The caller decides whether a positional
    # match is safe or an error should be raised.
    return None


def _is_dataframe(X):
    """Return True when ``X`` is a dataframe, whatever its column labels are."""
    return nw.dependencies.is_into_dataframe(X)


def _raise_if_labels_cannot_be_matched(X, reference):
    """Refuse to guess when a dataframe has no usable column names.

    When the model has recorded input signal names, a dataframe whose column
    labels are not all strings cannot be matched by name. Falling back to
    positional matching would silently use the wrong signal if the columns were
    reordered, so an explicit error is raised instead.
    """
    if reference is None or not _is_dataframe(X):
        return
    raise ValueError(
        "X is a dataframe, but its columns do not have string names, so they "
        "cannot be matched to the input signals of the NARX model. A positional "
        "match could silently use the wrong signal if the columns were "
        "reordered.\n"
        "Rename the columns to the input signal names of the model:\n"
        f"{_format_names(reference)}\n"
        "or pass the values as an array to match the input signals by position."
    )


def _check_names(names, source):
    """Validate a 1D sequence of unique string names."""
    names = np.asarray(names, dtype=object)
    if names.ndim == 0:
        raise TypeError(
            f"{source} should be a sequence of strings, but a single "
            f"{type(names.item()).__name__} was given."
        )
    names = column_or_1d(names, dtype=object, warn=True)
    if not all(isinstance(name, str) for name in names):
        raise TypeError(
            f"{source} should only contain strings, but got the name types "
            f"{[type(name).__name__ for name in names]}."
        )
    if len(set(names.tolist())) != len(names):
        duplicates = sorted({name for name in names if list(names).count(name) > 1})
        raise ValueError(
            f"{source} contains duplicate names. Each input signal must have "
            "a unique name, otherwise the input signals cannot be told apart.\n"
            f"{_format_names(duplicates)}"
        )
    return np.asarray(names, dtype=object)


def _select_columns(X, names):
    """Return the dataframe ``X`` with its columns ordered as ``names``."""
    frame = nw.from_native(X)
    return nw.to_native(frame.select(list(names)))


def _missing_columns(column_names, reference):
    """Return the input signals of the model that are missing in ``X``."""
    name_set = set(column_names.tolist())
    return [name for name in reference if name not in name_set]


def _get_recorded_names(model):
    """Return the input signal names recorded by ``model``, or None.

    ``feature_names_`` is used when available. For models fitted by an older
    version, which only recorded ``feature_names_in_``, that attribute is used
    as a fallback so that models loaded from disk can still be matched by name.
    """
    names = getattr(model, "feature_names_", None)
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    return names


def _resolve_fit_columns(X, declared_names, stored_names):
    """Match the columns of ``X`` to the input signals of the model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_columns)
        The training input data.

    declared_names : array-like of shape (n_input_signals,) or None
        The names passed to the ``feature_names`` parameter.

    stored_names : array-like of shape (n_input_signals,) or None
        The input signal names recorded by a previous :term:`fit`, if any.

    Returns
    -------
    X : array-like
        ``X`` with its columns ordered as the input signals of the model.
        Columns that do not correspond to an input signal are dropped.

    names : ndarray of shape (n_input_signals,), dtype=object or None
        The input signal names in feature index order, or None if the input
        signals are anonymous.
    """
    column_names = _get_column_names(X)

    if declared_names is not None:
        reference = _check_names(declared_names, "`feature_names`")
    elif stored_names is not None:
        # Refitting a model that has already recorded its input signals, e.g.,
        # a model loaded from disk, whose feature indices refer to the recorded
        # signal order. The columns are matched by name, like at predict time.
        reference = _check_names(stored_names, "`feature_names_`")
    elif column_names is not None:
        # Every column of X is an input signal, in the order of the columns.
        return X, column_names
    else:
        # X has no column names and no names were declared, so the input
        # signals are anonymous and are matched by position.
        return X, None

    if column_names is None:
        _raise_if_labels_cannot_be_matched(X, reference)
        n_columns = np.shape(X)[1] if np.ndim(X) == 2 else None
        if n_columns is not None and n_columns != len(reference):
            raise ValueError(
                f"The NARX model has {len(reference)} input signal(s), but X "
                f"has {n_columns} feature(s) without column names. X should "
                "have the same number of features as the input signals, or be "
                "a dataframe whose columns can be matched by name."
            )
        # X has no column names, so its columns are assumed to be in the order
        # of the input signals of the model.
        return X, reference

    missing = _missing_columns(column_names, reference)
    if missing:
        raise ValueError(
            "The columns of X do not match the input signals of the NARX model, "
            "so the model cannot know which signal each column refers to.\n"
            "Missing input columns:\n"
            f"{_format_names(missing)}\n"
            "The input signals of the model are:\n" + _format_names(reference)
        )
    if column_names.tolist() != reference.tolist():
        # Columns that do not belong to the model are ignored. The input
        # signals of the model are selected by name.
        X = _select_columns(X, reference)
    return X, reference


def _align_predict_columns(X, reference):
    """Order the columns of ``X`` according to the model input signals.

    Every input signal of the model must be present in ``X``. Columns that do
    not belong to the model are ignored.

    Parameters
    ----------
    X : array-like
        The input data. Inputs without string column names are returned
        unchanged and are assumed to be in the model input signal order.

    reference : array-like of shape (n_input_signals,) or None
        The input signal names of the model in feature index order.

    Returns
    -------
    X : array-like
        ``X`` with its columns ordered as ``reference``.
    """
    column_names = _get_column_names(X)
    if column_names is None:
        # X has no usable column names. A dataframe with non-string labels is
        # rejected when the model has recorded names, because a positional
        # match could be wrong. Arrays are matched positionally.
        _raise_if_labels_cannot_be_matched(X, reference)
        return X
    if reference is None:
        raise ValueError(
            "X has column names, but the NARX model was fitted without input "
            "signal names. The columns of X cannot be matched to the input "
            "signals of the model, so a reordered table would silently produce "
            "wrong predictions.\n"
            "Fit the model with a dataframe whose columns are all strings, or "
            "pass the input signal names via the `feature_names` parameter."
        )

    reference = _check_names(reference, "`feature_names_`")
    missing = _missing_columns(column_names, reference)
    if missing:
        raise ValueError(
            "The columns of X do not match the input signals of the NARX model, "
            "so the model cannot know which signal each column refers to.\n"
            "Missing input columns:\n"
            f"{_format_names(missing)}\n"
            "The input signals of the model (in the order used at fit time) "
            "are:\n" + _format_names(reference)
        )
    if column_names.tolist() != reference.tolist():
        X = _select_columns(X, reference)
    return X
