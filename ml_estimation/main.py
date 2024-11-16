# -----------------------------------------------------------------------------------------------------------------
# SCRIPT DESCRIPTION:
# This script simulates flight demand and estimates parameters for a logistic demand model using Maximum Likelihood
# Estimation (MLE). It generates synthetic data by drawing from Poisson distributions and calculating market shares.
# Key functions include partial derivatives of the likelihood function, optimization of parameters using SciPy's
# `minimize` function, and 3D plotting of the objective and log-likelihood surfaces. The script allows for multiple
# simulations of flight demand, with options to optimize parameters over various initial guesses to avoid local minima.
# Visualization functions help inspect the behavior of the objective function. The primary output is the optimized
# parameter estimates (alpha and delta) for the demand model.
# -----------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from ml_estimation.simulate_data import simFlightDemand
from ml_estimation.calc_likelihood import maxLike
from ml_estimation.calc_likelihood import objective

def main():
    """
    Main function to simulate flight demand and estimate model parameters.
    """
    # Initialize Parameters
    N = 1000      # Number of simulations
    J = 2         # Number of products (e.g., flights)
    T = 90        # Number of time periods
    delta = 0.5   # True delta parameter
    alpha = 0.03  # True alpha parameter
    lambd = 30    # Arrival rate parameter
    # Simulate demand for N flights
    df = pd.DataFrame()
    for i in range(N):
        df_new = simFlightDemand(J, T, delta, alpha, lambd)
        df_new["sim"] = i  # Simulation identifier
        df = pd.concat([df, df_new], ignore_index=True)
    # Optimization to estimate parameters
    min_guess = np.array([1, 1])
    # Loop over multiple initial guesses to avoid local minima
    for i in range(50):
        result = maxLike(df)
        # Update min_guess if a better solution is found
        if objective(result.x, df) < objective(min_guess, df):
            min_guess = result.x
    # Output the estimated parameters
    print(f"Estimated alpha: {min_guess[0]:.4f}")
    print(f"Estimated delta: {min_guess[1]:.4f}")
    return min_guess

if __name__ == "__main__":
    estimated_params = main()
