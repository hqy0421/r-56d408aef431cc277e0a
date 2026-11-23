.. _narx:

=======================================================
Reduced polynomial NARX model for system identification
=======================================================

.. currentmodule:: fastcan.narx

:class:`NARX` is a class to build a reduced polynomial NARX (Nonlinear AutoRegressive eXogenous) model for system identification.
The structure of a polynomial NARX model is shown below:

.. math::
    y(k) = 0.5y(k-1) + 0.3u_0(k)^2 + 2u_0(k-1)u_0(k-3) + 1.5u_0(k-2)u_1(k-3) + 1

where :math:`k` is the time index, :math:`u_0` and :math:`u_1` are input signals, and :math:`y` is the output signal.


The NARX model is composed of three parts:

#. Terms: :math:`y(k-1)`, :math:`u_0(k)^2`, :math:`u_0(k-1)u_0(k-3)`, and :math:`u_0(k-2)u_1(k-3)`
#. Coefficients: 0.5, 0.3, 2, and 1.5
#. Intercept: 1


To build a reduced polynomial NARX model, we normally need the following steps:

#. Build term library
    #. Get time shifted variables: e.g., :math:`u(t-2)` and :math:`y(t-1)`
    #. Get terms by nonlinearising the time shifted variables with polynomial basis, e.g., :math:`u_0(k-1)u_0(k-3)` and :math:`u_0(k-2)u_1(k-3)`
#. Search useful terms by feature selection, i.e., `Reduced Order Modelling (ROM) <https://www.mathworks.com/discovery/reduced-order-modeling.html>`_
#. Learn the coefficients by least-squares

.. rubric:: Examples

* See :ref:`sphx_glr_auto_examples_plot_narx.py` for an example of the basic usage of the NARX model.

Input signal names and archived models
======================================

The input signals of a NARX model are identified by their position in `X`, i.e.,
the feature ids in `feat_ids` and `delay_ids` refer to column indices.
This is fragile when a model is archived and later applied to data whose columns
are ordered differently, e.g., a data-collection table that was reorganized
between measurement campaigns.
Applying the model positionally would silently use the wrong signal for a term
and return a plausible, but wrong, prediction.

To avoid this, :class:`NARX` matches the columns of `X` to the input signals
by name when the names are known.
The names are taken from the column names of `X` when `X` is a dataframe whose
columns are all strings, or from the `feature_names` parameter.
They are stored in `feature_names_` and reused by later calls to
:meth:`NARX.fit` and :meth:`NARX.predict`, including after the model has been
saved and loaded (e.g., with pickle or joblib).

.. code-block:: python

    >>> import numpy as np
    >>> import pandas as pd
    >>> from fastcan.narx import NARX
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((100, 2))
    >>> y = X[:, 0] - 2 * X[:, 1]
    >>> # The names are recorded when the model is fitted on a dataframe.
    >>> narx = NARX(feat_ids=[[-1, 0], [-1, 1]], delay_ids=[[-1, 0], [-1, 0]])
    >>> narx.fit(pd.DataFrame(X, columns=["position", "velocity"]), y)
    NARX(delay_ids=[[-1, 0], [-1, 0]], feat_ids=[[-1, 0], [-1, 1]])
    >>> narx.feature_names_
    array(['position', 'velocity'], dtype=object)
    >>> # A later table has the columns in a different order: the model still
    >>> # uses the same signals, so the prediction is unchanged.
    >>> y_hat = narx.predict(pd.DataFrame(X, columns=["position", "velocity"]))
    >>> np.allclose(
    ...     narx.predict(pd.DataFrame(X[:, ::-1], columns=["velocity", "position"])),
    ...     y_hat,
    ... )
    True

The matching rules are:

* Columns are matched by name, so the column order of `X` does not matter.
* Columns of `X` that do not belong to the model are ignored.
* A missing input column raises a :class:`ValueError` that lists the missing
  input signals, instead of returning a prediction.
* Duplicate column names in `X` raise a :class:`ValueError`, because the signal
  that a column refers to is ambiguous.
* A dataframe whose column labels are not all strings cannot be matched by
  name. When the model has recorded input signal names, passing such a
  dataframe raises a :class:`ValueError` instead of silently falling back to
  positional matching. Pass the values as an array to match by position.
* A model fitted without input signal names raises a :class:`ValueError` when
  :meth:`NARX.predict` is called with a named dataframe, rather than falling
  back to positional matching. Pass an array to use positional matching, or
  refit the model with named data or with `feature_names`.

Use :func:`print_narx` to check which signals and delays a loaded model uses.
When the model has recorded names, the terms are printed with the signal names,
e.g., ``u2[k-3]``, instead of the column indices, e.g., ``X[k-3,2]``.

Multiple time series and missing values
=======================================

For time series modelling, the continuity of the data is important.
The continuity means all neighbouring samples have the same time interval.
A discontinuous time series can be treated as multiple pieces of continuous time series.
The discontinuity can be caused by

#. Different measurement sessions, e.g., one measurement session is carried out at Monday and
   lasts for one hour at 1 Hz, and another measurement is carried out at Tuesday and lasts for two hours at 1 Hz
#. Missing values

As a result, there are two ways to inform :class:`NARX` the training data from different measurement sessions.
From version 0.5, the difference measurement sessions can be set by `session_sizes` in :meth:`NARX.fit` and :func:`make_narx`.
Alternatively, different measurement sessions can also be notified by inserting `np.nan` to break the data.
The following examples show how to notify :class:`NARX` the training data from different measurement sessions.

.. code-block:: python

    >>> import numpy as np
    >>> x0 = np.zeros((3, 2)) # First measurement session
    >>> x1 = np.ones((5, 2)) # Second measurement session
    >>> max_delay = 2 # Assume the maximum delay for NARX model is 2
    >>> # Insert (at least max_delay number of) np.nan to break the two measurement sessions
    >>> u = np.r_[x0, [[np.nan, np.nan]]*max_delay, x1]
    >>> # Or use session_sizes to break the two measurement sessions
    >>> session_sizes = [3, 5]

.. note::
    It is important to separate different measurement sessions using either method.
    Otherwise, the model will incorrectly assume that the time interval between two measurement sessions
    is the same as the sampling interval within a session.

    Actually, the two methods will result in the exact same models when `X` is not None or multiple-step-ahead prediction is not used for training.
    If `X` is None, i.e. Auto-Regression models, and multiple-step-ahead prediction is used for training, only `session_sizes` can
    separate the two measurement sessions.
    Because when using multiple-step-ahead prediction, the optimiser relies on the missing values in inputs `X` rather than output `y`, while Auto-Regression models do not have input `X`.


One-step-ahead and multiple-step-ahead prediction
=================================================

In system identification, two types of predictions can be made by NARX models:

#. One-step-ahead prediction: predict the output at the next time step given the measured input and the measured output at the past time steps, e.g., :math:`\hat{y}(k) = 0.5y(k-1) + 0.3u_0(k-1)^2`, where :math:`\hat{y}(k)` is the predicted output at time step :math:`k`
#. Multiple-step-ahead prediction: predict the output at the next time step given the measured input and the predicted output at the past time steps, e.g., :math:`\hat{y}(k) = 0.5\hat{y}(k-1) + 0.3u_0(k-1)^2`

Obviously, the multiple-step-ahead prediction is more challenging, because the predicted output is used as the input to predict the output at the next step, which may accumulate the error.
The `prediction` method of :class:`NARX` gives the multiple-step-ahead prediction.

It should also be noted the different types of predictions in model training.

#. If assume the NARX model is one-step-ahead prediction structure, we know all input data for training in advance,
   and the model can be trained by the simple ordinary least-squares (OLS) method
#. If assume the NARX model is a multiple-step-ahead prediction structure, the input data, like :math:`\hat{y}(k-1)` is
   unknown in advance. Therefore, the training data must first be generated by the multiple-step-ahead prediction with
   the initial model coefficients, and then the coefficients can be updated recursively

ARX and OE model
----------------

To better understand the two types of training, it is helpful to know two linear time series model structures,
i.e., `ARX (AutoRegressive eXogenous) model <https://www.mathworks.com/help/ident/ref/arx.html>`_ and
`OE (output error) model <https://www.mathworks.com/help/ident/ref/oe.html>`_.

The ARX model structure is

.. math::
    (1-A(z))y(t) = B(z)u(t) + e(t)

where :math:`A(z)` and :math:`B(z)` are polynomials of :math:`z`, e.g.,
:math:`A(z) = 1.5 z^{-1} - 0.7 z^{-2}`, :math:`z` is a complex number from `Z-transformation <https://en.wikipedia.org/wiki/Z-transform>`_
and :math:`z^{-1}` can be view as a time-shift operator, and :math:`e(t)` is the white noise.

The OE model structure is

.. math::
    y(t) = \frac{B(z)}{(1-A(z))} u(t) + e(t)


The ARX model and OE model is very similar. If we rewrite ARX model to :math:`y(t) = \frac{B(z)}{(1-A(z))} u(t) + \frac{1}{(1-A(z))} e(t)`,
we can see that both the ARX model and the OE model have the same transfer function :math:`\frac{B(z)}{(1-A(z))}`.
The two key differences are:

#. noise assumption
#. `best linear unbiased estimator (BLUE) <https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem>`_

For noise assumption, the noise of the ARX model passes through a filter :math:`\frac{1}{(1-A(z))}` while the noise of the OE model not.
The structure of ARX models implies that the measurement noise is a very special colored noise, which is not common in practice.

For difference in BLUE, to get BLUE of models, we need to rewrite models to the form of

.. math::
    e(t) = y(t) - \hat{y}(t)

where :math:`\hat{y}(t)` is a predictor, which describes how to compute the predicted output to make the error white.
It is easy to find that for a ARX model

.. math::
    \hat{y}(t) = A(z)y(t) + B(z)u(t)

while for an OE model

.. math::
    \hat{y}(t) = \frac{B(z)}{(1-A(z))} u(t) = A(z)\hat{y}(t) + B(z)u(t)

The difference between the two predictors is the one for ARX models uses both measured input and output to make prediction,
while the one for OE models only uses measured input to make prediction.
In other words, the predictor of the BLUE for ARX computes one-step-ahead prediction,
while the predictor of the BLUE for OE computes multiple-step-ahead prediction.

In summary, using multiple-step-ahead prediction in model training is more aligned with the real noise condition, but harder to train.
However, using one-step-ahead prediction in model training is easier to train, but the noise assumption is too strong to be true.
The `fit` method of :class:`NARX` uses the one-step-ahead prediction for training by default,
but the multiple-step-ahead prediction can be used by setting the parameter `coef_init`.

.. rubric:: Examples

* See :ref:`sphx_glr_auto_examples_plot_narx_msa.py` for an illustration of the multiple-step-ahead NARX model.
