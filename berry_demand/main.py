# -----------------------------------------------------------------------------------------------------------------
# SCRIPT DESCRIPTION:
# This script models demand estimation for multiple firms over several time periods using both IV (Instrumental 
# Variable) and OLS (Ordinary Least Squares) regression. It generates synthetic data with correlated errors to simulate 
# product demand and prices. The estimation function, `estimModel`, calculates market shares and uses IV to instrument 
# prices with input prices. Demand draws are sampled from three distributions (custom discrete, cumulative sum, 
# and uniform), and the multiprocessing library is used to parallelize the estimation process for faster computation.
# The results are printed along with the total execution time for performance tracking.
# -----------------------------------------------------------------------------------------------------------------

# main.py

import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from berry_demand.data_simulation import simulate_data
from berry_demand.demand_draws import draw_demand_custom, draw_demand_cumsum, draw_demand_uniform
from berry_demand.estimation import estimModel

if __name__ == '__main__':
    t0 = time.time()

    # Parameters
    N = 10000      # Number of consumers
    J = 3          # Number of products (including outside good)
    T = 365        # Number of time periods

    # True parameters for simulation
    beta0 = 0
    beta1 = 0.5
    alpha = 0.18
    omega0 = 1
    omega1 = 0.5
    rho = 0.9      # Correlation between error terms

    # Simulate data
    df = simulate_data(N, J, T, beta0, beta1, alpha, omega0, omega1, rho)

    # Draw demand using different methods
    X_custm = draw_demand_custom(df, N, J, T)
    X_cumsum = draw_demand_cumsum(df, N, J, T)
    X_unif = draw_demand_uniform(df, N, J, T)

    # Prepare arguments for estimation
    args_list = [
        (X_custm, df.copy(), N, J, T),
        (X_cumsum, df.copy(), N, J, T),
        (X_unif, df.copy(), N, J, T)
    ]

    # Run estimation in parallel
    with Pool(processes=3) as pool:
        results = pool.map(estimModel, args_list)

    # Close the pool
    pool.close()
    pool.join()

    # Print results
    print("Estimation Results (beta1_hat, alpha_hat):")
    demand_methods = ['Custom Discrete', 'Cumulative Sum', 'Uniform']
    for method, res in zip(demand_methods, results):
        print(f"{method} Demand Draw: beta1_hat = {res[0]:.4f}, alpha_hat = {res[1]:.4f}")

    # Total execution time
    t1 = time.time()
    total = t1 - t0
    print(f"Total Execution Time: {total:.2f} seconds")
