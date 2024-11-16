
import numpy as np
import matplotlib.pyplot as plt
from likelihood_functions import objective, loglikelihood

def graph_objective(df):
    """
    Generates a 3D surface plot of the objective function over a grid of alpha and delta values.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    """
    a_vals = np.linspace(0, 1, 100)
    d_vals = np.linspace(0, 1, 100)
    Z = np.zeros((a_vals.shape[0], d_vals.shape[0]))
    # Calculate objective values for each point on the grid
    for i in range(len(a_vals)):
        for j in range(len(d_vals)):
            Z[i, j] = objective([a_vals[i], d_vals[j]], df)
    # Create a 3D surface plot
    x = np.outer(a_vals, np.ones(a_vals.shape[0]))
    y = x.copy().T
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, Z, cmap='viridis', edgecolor='green')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Delta')
    ax.set_zlabel('Objective Function Value')
    plt.show()

def graph_loglikelihood(df):
    """
    Generates a 3D surface plot of the log-likelihood function over a grid of alpha and delta values.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    """
    a_vals = np.linspace(0, 1, 100)
    d_vals = np.linspace(0, 1, 100)
    Z = np.zeros((a_vals.shape[0], d_vals.shape[0]))
    # Calculate log-likelihood values for each point on the grid
    for i in range(len(a_vals)):
        for j in range(len(d_vals)):
            Z[i, j] = loglikelihood([a_vals[i], d_vals[j]], df)
    # Create a 3D surface plot
    x = np.outer(a_vals, np.ones(a_vals.shape[0]))
    y = x.copy().T
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, Z, cmap='viridis', edgecolor='green')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Delta')
    ax.set_zlabel('Negative Log-Likelihood')
    plt.show()
