import pandas as pd
import numpy as np


class GroupEstimate(object):
    """
    A group-based estimator that predicts continuous values
    from categorical input data.

    Parameters
    ----------
    estimate : str
        The estimation method to use. Either 'mean' or 'median'.
    """

    def __init__(self, estimate='mean'):
        if estimate not in ('mean', 'median'):
            raise ValueError("estimate must be 'mean' or 'median'.")
        self.estimate = estimate
        self._group_estimates = None
        self._columns = None

    def fit(self, X, y, default_category=None):
        """
        Fit the model by computing group estimates from X and y.

        Parameters
        ----------
        X : pd.DataFrame
            Categorical input features.
        y : array-like
            Continuous target values (no missing values).
        default_category : str, optional
            Column name to fall back on when a combination is missing.
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y, name='__y__').reset_index(drop=True)

        self._columns = list(X.columns)
        self._default_category = default_category

        df = pd.concat([X, y], axis=1)

        if self.estimate == 'mean':
            self._group_estimates = df.groupby(self._columns)['__y__'].mean()
        else:
            self._group_estimates = df.groupby(self._columns)['__y__'].median()

        # Bonus: fit fallback estimates on default_category column
        if default_category is not None:
            if self.estimate == 'mean':
                self._default_estimates = df.groupby(default_category)['__y__'].mean()
            else:
                self._default_estimates = df.groupby(default_category)['__y__'].median()
        else:
            self._default_estimates = None

        return self

    def predict(self, X_):
        """
        Predict estimates for new observations based on group membership.

        Parameters
        ----------
        X_ : array-like or pd.DataFrame
            New observations with the same columns as X used in fit().

        Returns
        -------
        np.ndarray
            Predicted estimates; NaN for unrecognized groups.
        """
        X_ = pd.DataFrame(X_, columns=self._columns)
        results = []
        missing_count = 0

        for _, row in X_.iterrows():
            key = tuple(row[col] for col in self._columns)
            key = key[0] if len(key) == 1 else key

            if key in self._group_estimates.index:
                results.append(self._group_estimates[key])
            else:
                # Bonus: try default_category fallback
                if self._default_estimates is not None:
                    fallback_key = row[self._default_category]
                    if fallback_key in self._default_estimates.index:
                        results.append(self._default_estimates[fallback_key])
                    else:
                        results.append(np.nan)
                        missing_count += 1
                else:
                    results.append(np.nan)
                    missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} observation(s) had missing groups and were assigned NaN.")

        return np.array(results)
