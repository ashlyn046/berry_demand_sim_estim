# demand_draws.py

import numpy as np
import pandas as pd
from scipy import stats

def draw_demand_custom(df, N, J, T):
    """
    Draws demand using a custom discrete distribution.

    Parameters:
    - df (DataFrame): Data containing market shares.
    - N (int): Number of consumers.
    - J (int): Number of products.
    - T (int): Number of time periods.

    Returns:
    - X_custm (list): List of arrays containing demand counts for each product in each time period.
    """
    X_custm = []
    for idx in range(T):
        pk = df.loc[df.t == idx].sj.values
        xk = np.arange(J)
        custm = stats.rv_discrete(name='custm', values=(xk, pk))
        Q = custm.rvs(size=N)
        Q_counts = np.bincount(Q, minlength=J)
        X_custm.append(Q_counts)
    return X_custm

def draw_demand_cumsum(df, N, J, T):
    """
    Draws demand using cumulative sum distribution.

    Parameters:
    - df (DataFrame): Data containing market shares.
    - N (int): Number of consumers.
    - J (int): Number of products.
    - T (int): Number of time periods.

    Returns:
    - X_cumsum (list): List of arrays containing demand counts for each product in each time period.
    """
    X_cumsum = []
    for idx in range(T):
        pk = df.loc[df.t == idx].sj.values
        Q = np.searchsorted(np.cumsum(pk), np.random.random(N))
        Q_counts = np.bincount(Q, minlength=J)
        X_cumsum.append(Q_counts)
    return X_cumsum

def draw_demand_uniform(df, N, J, T):
    """
    Draws demand using uniform distribution.

    Parameters:
    - df (DataFrame): Data containing market shares.
    - N (int): Number of consumers.
    - J (int): Number of products.
    - T (int): Number of time periods.

    Returns:
    - X_unif (list): List of arrays containing demand counts for each product in each time period.
    """
    X_unif = []
    for idx in range(T):
        pk = df.loc[df.t == idx].sj.values
        Q = np.searchsorted(np.cumsum(pk), np.random.uniform(size=N))
        Q_counts = np.bincount(Q, minlength=J)
        X_unif.append(Q_counts)
    return X_unif
