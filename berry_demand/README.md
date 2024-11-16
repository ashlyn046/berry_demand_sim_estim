# Berry Demand Estimation Project

## Overview
This project simulates product demand and prices over several time periods, estimates market shares, and calculates demand parameters using synthetic data. The estimation process employs Instrumental Variable (IV) and Ordinary Least Squares (OLS) regression, leveraging multiprocessing for efficient computation.

## Features
- **Data Simulation**: Generates synthetic data for demand estimation, including product characteristics, prices, and market shares.
- **Demand Sampling Methods**: Provides three approaches for drawing demand:
  - Custom discrete distribution
  - Cumulative sum distribution
  - Uniform distribution
- **Demand Estimation**: Estimates price sensitivity and product characteristic effects using IV regression.
- **Efficiency**: Utilizes Python's multiprocessing library to parallelize computations across demand sampling methods.

## File Descriptions

### `data_simulation.py`
Simulates synthetic data for demand estimation:
- **Function**: `simulate_data(N, J, T, beta0, beta1, alpha, omega0, omega1, rho)`
- **Processes**:
    - Generates product characteristics (`x`) and input prices (`z`).
    - Models correlated error terms in price and utility equations.
    - Computes market shares using a logit model.

### `demand_draws.py`
Implements three demand sampling methods:
1. **Custom Discrete**: Matches market share probabilities.
    - **Function**: `draw_demand_custom(df, N, J, T)`
2. **Cumulative Sum**: Uses cumulative probabilities for sampling.
    - **Function**: `draw_demand_cumsum(df, N, J, T)`
3. **Uniform**: Generates demand draws with uniform random samples.
    - **Function**: `draw_demand_uniform(df, N, J, T)`

### `estimation.py`
Defines the estimation function using IV regression:
- **Function**: `estimModel(args)`
- **Processes**:
    - Calculates observed market shares and deltas.
    - Performs IV regression with input prices as instruments to estimate:
        - `beta1` (product characteristic sensitivity).
        - `alpha` (price sensitivity).

### `main.py`
Main script that orchestrates the entire process:
- **Processes**:
    1. Data simulation using `data_simulation.py`.
    2. Demand sampling using methods from `demand_draws.py`.
    3. Demand estimation using `estimation.py` in parallel processes.
- **Functionality**:
    - Measures total execution time.
    - Prints estimation results for each demand drawing method.

## Notes
- **Parallel Processing**: Three parallel processes handle demand estimation for the three sampling methods, significantly reducing runtime.
- **Robustness Testing**: Compare results across sampling methods to evaluate the robustness of demand estimation.

For more details, refer to the inline comments in each script.

## References

Berry, S., Levinsohn, J., & Pakes, A. (1995). *Automobile Prices in Market Equilibrium*. *Econometrica*, 63(4), 841–890. [https://www.econometricsociety.org/publications/econometrica/1995/07/01/automobile-prices-market-equilibrium](https://www.econometricsociety.org/publications/econometrica/1995/07/01/automobile-prices-market-equilibrium)
