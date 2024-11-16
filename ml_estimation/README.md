# Flight Demand Estimation Project

## Overview
This project simulates flight demand and estimates parameters for a logistic demand model using Maximum Likelihood Estimation (MLE). It generates synthetic data by drawing from Poisson distributions and calculating market shares. The estimation process involves computing partial derivatives of the likelihood function, optimizing parameters using SciPy's `minimize` function, and visualizing the objective and log-likelihood surfaces through 3D plots. The project supports multiple simulations with varied initial guesses to ensure robust parameter estimation.

## Features
- **Data Simulation**: Generates synthetic flight demand data based on specified parameters.
- **Maximum Likelihood Estimation (MLE)**: Estimates model parameters (`alpha` and `delta`) by optimizing the likelihood function.
- **Partial Derivatives**: Calculates derivatives of the likelihood function with respect to parameters to facilitate optimization.
- **Optimization**: Utilizes SciPy's `minimize` function with multiple initial guesses to avoid local minima.
- **Visualization**: Creates 3D surface plots of the objective function and log-likelihood to analyze the estimation landscape.
- **Multiple Simulations**: Supports running numerous simulations to enhance the reliability of parameter estimates.

## Outputs
- **Estimated Parameters**:
  - `alpha`: Sensitivity parameter in the logistic demand model.
  - `delta`: Delta parameter in the logistic demand model.

- **Execution Time**: Total runtime for simulation and estimation processes.

- **Visualizations**:
- **Objective Function Surface**: 3D plot showing the landscape of the objective function over a range of `alpha` and `delta` values.
- **Log-Likelihood Surface**: 3D plot illustrating the log-likelihood values across different parameter combinations.

Example Visualization:
![Log-Likelihood Surface](path_to_your_image.png)

## File Descriptions

### `main.py`
**Description**: Orchestrates the simulation of flight demand and estimation of model parameters.
- **Functions**:
- `main()`: Simulates flight demand for `N` simulations, performs MLE to estimate `alpha` and `delta`, and prints the estimated parameters.

- **Process**:
1. Initializes simulation parameters.
2. Simulates demand using `simFlightDemand` from `simulate_data.py`.
3. Performs optimization to estimate parameters using `maxLike` from `calc_likelihood.py`.
4. Prints the estimated `alpha` and `delta`.

### `calc_likelihood.py`
**Description**: Contains functions to compute the likelihood, its partial derivatives, and perform optimization for MLE.
- **Functions**:
- `dalpha(a, d, df)`: Partial derivative of the likelihood with respect to `alpha`.
- `ddelta(a, d, df)`: Partial derivative of the likelihood with respect to `delta`.
- `loglikelihood(params, df)`: Computes the negative log-likelihood for given parameters.
- `comp_grad(params, df)`: Computes the gradient vector of the likelihood function.
- `objective(params, df)`: Objective function to minimize (sum of squared partial derivatives).
- `mle(df)`: Performs MLE by minimizing the negative log-likelihood using SciPy's `minimize`.

### `simulate_data.py`
**Description**: Generates synthetic flight demand data based on a logistic demand model.
- **Functions**:
- `simFlightDemand(J, T, delta, alpha, lambd)`: Simulates flight demand for `J` products over `T` time periods.

- **Process**:
1. Generates random prices for each product and time period.
2. Calculates market shares using the logistic model.
3. Simulates demand using a Poisson distribution based on market shares and arrival rates.
4. Merges demand and arrival data into a comprehensive DataFrame.

### `visualizations.py`
**Description**: Provides functions to visualize the objective and log-likelihood functions.
- **Functions**:
- `graph_objective(df)`: Generates a 3D surface plot of the objective function over a grid of `alpha` and `delta` values.
- `graph_loglikelihood(df)`: Generates a 3D surface plot of the log-likelihood function over a grid of `alpha` and `delta` values.

- **Usage**:
- Run the script to generate and display the plots, which help in analyzing the behavior of the estimation process.

## Notes
- **Extensibility**: The modular structure allows for easy addition of new simulation or estimation methods.
- **Optimization Robustness**: Running multiple initial guesses in the optimization process helps avoid convergence to local minima, enhancing the reliability of parameter estimates.
- **Visualization**: 3D plots assist in understanding the optimization landscape and diagnosing potential issues with the estimation process.
- **Performance**: For large-scale simulations, consider optimizing the code further or utilizing parallel processing techniques.

For further details and customization, refer to the inline comments within each script.
