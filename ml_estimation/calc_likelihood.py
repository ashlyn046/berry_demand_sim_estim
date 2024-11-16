
import numpy as np
from scipy.optimize import minimize

#-----------------------------------------------------------PARTIAL DERIVATIVES-----------------------------------------------------------
def dalpha(a, d, df):
    """
    Partial derivative of the likelihood function with respect to alpha.

    Parameters:
    - a (float): Current estimate of alpha.
    - d (float): Current estimate of delta.
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - float: Sum of partial derivatives with respect to alpha.
    """
    # Calculate the partial derivative for each observation
    pderiv = -((df.p * np.exp(df.p * a)) * (df.Q * np.exp(df.p * a) + np.exp(d) * df.Q - np.exp(d) * df.lam)) / \
             np.square(np.exp(df.p * a) + np.exp(d))
    return np.sum(pderiv)

def ddelta(a, d, df):
    """
    Partial derivative of the likelihood function with respect to delta.

    Parameters:
    - a (float): Current estimate of alpha.
    - d (float): Current estimate of delta.
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - float: Sum of partial derivatives with respect to delta.
    """
    # Calculate the partial derivative for each observation
    pderiv = (np.exp(a * df.p) * ((df.Q - df.lam) * np.exp(d) + np.exp(a * df.p) * df.Q)) / \
             np.square(np.exp(d) + np.exp(a * df.p))
    return np.sum(pderiv)

def loglikelihood(params, df):
    """
    Computes the negative log-likelihood of the logistic demand model.

    Parameters:
    - params (array): Array containing alpha and delta.
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - float: Negative sum of the log-likelihood values.
    """
    a, d = params
    # Precompute factorials for Q (demand quantities)
    q_fact = np.zeros(df.Q.shape[0])
    for i in range(df.Q.shape[0]):
        q_fact[i] = np.math.factorial(int(round(df.Q.iloc[i])))
    # Calculate the log-likelihood for each observation
    loglik = (-df.lam * np.exp(d - a * df.p)) / (1 + np.exp(d - a * df.p)) + \
             df.Q * np.log(df.lam) + df.Q * (d - a * df.p) - \
             df.Q * np.log(1 + np.exp(d - a * df.p)) - np.log(q_fact)
    return -np.sum(loglik)  # Return negative log-likelihood for minimization

def comp_grad(params, df):
    """
    Computes the gradient vector of the likelihood function.

    Parameters:
    - params (array): Array containing alpha and delta.
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - array: Gradient vector [dalpha, ddelta].
    """
    a, d = params
    return np.array([dalpha(a, d, df), ddelta(a, d, df)])
#----------------------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------MAIN FUNCTIONALITY-----------------------------------------------------------
def objective(params, df):
    """
    Objective function to minimize: sum of squared partial derivatives.

    Parameters:
    - params (array): Array containing alpha and delta.
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - float: Sum of squared partial derivatives.
    """
    a, d = params
    return np.square(dalpha(a, d, df)) + np.square(ddelta(a, d, df))

def mle(df):
    """
    Performs Maximum Likelihood Estimation to find optimal alpha and delta.

    Parameters:
    - df (DataFrame): DataFrame containing the data.

    Returns:
    - result (OptimizeResult): The optimization result returned by SciPy's minimize function.
    """
    # Random initial guesses for alpha and delta
    x_init = np.random.uniform(low=0, high=1)
    y_init = np.random.uniform(low=0, high=1)
    initial_guess = np.array([x_init, y_init])
    # Minimize the negative log-likelihood function
    result = minimize(
        fun=loglikelihood,
        x0=initial_guess,
        args=(df,),
        method='L-BFGS-B',
        bounds=((0.0, 1.0), (0.0, 1.0)),
        options={'gtol': 1e-15, 'ftol': 1e-15}
    )
    return result
#----------------------------------------------------------------------------------------------------------------------------------------
