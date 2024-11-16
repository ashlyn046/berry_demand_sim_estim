# data_simulation.py

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

def simulate_data(N, J, T, beta0, beta1, alpha, omega0, omega1, rho):
    """
    Simulates data for demand estimation.

    Parameters:
    - N (int): Number of consumers
    - J (int): Number of products (including outside good)
    - T (int): Number of time periods
    - beta0 (float): Intercept in utility model
    - beta1 (float): Coefficient for product characteristic x in utility model
    - alpha (float): Price sensitivity parameter in utility model
    - omega0 (float): Intercept in price equation
    - omega1 (float): Coefficient for input prices in price equation
    - rho (float): Correlation between error terms in utility and price equations

    Returns:
    - df (DataFrame): Simulated data containing product characteristics, prices, market shares, etc.
    """
    # Generate product characteristics x and input prices z
    x = np.random.normal(size=J * T)
    z = np.random.normal(size=J * T)

    # Generate correlated error terms nu (price equation) and xi (utility equation)
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    error_terms = multivariate_normal.rvs(mean, cov, size=J * T)
    nu = error_terms[:, 0] / 2
    xi = error_terms[:, 1] / 15

    # Generate prices p
    p = omega0 - omega1 * z + nu

    # Generate input prices (without nu)
    input_prices = omega0 - omega1 * z

    # Create DataFrame with data
    data = {
        'x': x,
        'xi': xi,
        'p': p,
        'input_prices': input_prices
    }
    df = pd.DataFrame(data)

    # Calculate delta1 (mean utility)
    df["delta1"] = beta0 + beta1 * df.x - alpha * df.p + df.xi

    # Create identifiers for time periods and products
    df["t"] = np.repeat(np.arange(T), J)
    df["j"] = np.tile(np.arange(J), T)

    # Set delta1 to 0 wherever j=0 (outside option)
    df.loc[df.j == 0, "delta1"] = 0

    # Calculate exponentiated delta
    df["edelta"] = np.exp(df.delta1)

    # Calculate market shares (logit model)
    df["sj"] = df.edelta / df.groupby("t").edelta.transform("sum")

    return df
