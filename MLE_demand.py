# -----------------------------------------------------------------------------------------------------------------
# SCRIPT DESCRIPTION:
# This script simulates flight demand and estimates parameters for a logistic demand model using maximum likelihood 
# estimation (MLE). It generates synthetic data by drawing from Poisson distributions and calculating market shares. 
# Key functions include partial derivatives of the likelihood function, optimization of parameters using SciPy's 
# `minimize` function, and 3D plotting of the objective and log-likelihood surfaces. The script allows for multiple 
# simulations of flight demand, with options to optimize parameters over various initial guesses to avoid local minima.
# Visualization functions help inspect the behavior of the objective function. The primary output is the optimized 
# parameter estimates (alpha and delta) for the demand model.
# -------

import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def dalpha(a, d, df):
	# partial derivative of liklihood function wrt alpha
	pderiv = -((df.p*np.exp(df.p*a))*(df.Q*np.exp(df.p*a) + np.exp(d)*df.Q - np.exp(d)*df.lam))/np.square(np.exp(df.p*a) + np.exp(d))
	return np.sum(pderiv)

def ddelta(a, d, df):
	# partial derivative of liklihood function wrt delta
	pderiv = (np.exp(a*df.p)*((df.Q - df.lam)*np.exp(d) + np.exp(a*df.p)*df.Q))/np.square(np.exp(d) + np.exp(a*df.p))
	return np.sum(pderiv)

def loglikelihood(params,df):
	a, d= params
	q_fact = np.zeros((df.Q.shape[0]))
	for i in range(df.Q.shape[0]):
		q_fact[i] = np.math.factorial(round(df.Q[i]))
	#adding a small count to qfact to avoid taaking log(0)
	#q_fact[q_fact == 0] = .0001
	loglik = (-df.lam*np.exp(d-a*df.p))/(1+np.exp(d-a*df.p)) + df.Q*np.log(df.lam) + df.Q*(d -a*df.p) - df.Q*np.log(1+np.exp(d - a*df.p)) - np.log(q_fact)
	return -np.sum(loglik)

def comp_grad(params,df):
	a, d= params
	return np.array([dalpha(a, d, df),ddelta(a,d,df)])

# Define a function that computes the sum of squared differences
def objective(params, df):
	# minimizing sum of squared pds
	a, d= params
	return np.square(dalpha(a, d, df)) + np.square(ddelta(a, d, df))

# IGNORE. This funciton was useful for checking that the actual mins were right according to the way we generated the data
def calcMinIdx(df):
	# Generate x values
	a_vals = np.linspace(0, 1, 100)
	a_vals = np.append(a_vals, alpha)
	d_vals = np.linspace(0,1,100)
	d_vals = np.append(d_vals, delta)
	return_vals = np.zeros((a_vals.shape[0],d_vals.shape[0]))
	for i in range(a_vals.shape[0]):
		for j in range(d_vals.shape[0]):
			return_vals[i,j] = objective(np.array([a_vals[i], d_vals[j]]), df)
	index = np.unravel_index(np.argmin(return_vals), return_vals.shape)
	return np.array([a_vals[index[0]], d_vals[index[1]]])

def graph_objective(df):
	a_vals = np.linspace(0, 1, 100)
	d_vals = np.linspace(0, 1, 100)
	Z = np.zeros((a_vals.shape[0], d_vals.shape[0]))
	#
	# Calculate objective values for each point on the grid
	for i in range(len(a_vals)):
	    for j in range(len(d_vals)):
	        Z[i, j] = objective([a_vals[i], d_vals[j]], df)
	#
	#create a three dimensional map where the XY plane is a grid 0 to 1 by o to 1 and the y axis is the value of Y for each point in grid
	x = np.outer(a_vals, np.ones(a_vals.shape[0]))   
	y = x.copy().T
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot_surface(x, y, Z, cmap='viridis',edgecolor='green')
	plt.show()

def graph_loglikelihood(df):
	a_vals = np.linspace(0, 1, 100)
	d_vals = np.linspace(0, 1, 100)
	Z = np.zeros((a_vals.shape[0], d_vals.shape[0]))
	#
	# Calculate objective values for each point on the grid
	for i in range(len(a_vals)):
	    for j in range(len(d_vals)):
	        Z[i, j] = loglikelihood([a_vals[i], d_vals[j]], df)
	#
	#create a three dimensional map where the XY plane is a grid 0 to 1 by o to 1 and the y axis is the value of Y for each point in grid
	x = np.outer(a_vals, np.ones(a_vals.shape[0]))   
	y = x.copy().T
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot_surface(x, y, Z, cmap='viridis',edgecolor='green')
	plt.show()

def simFlightDemand(J,T,delta,alpha,lambd):
	p = np.random.normal(loc=50,size = J * T)
	#
	data = {
		'p': p,
	}
	#
	df = pd.DataFrame(data)
	#
	#Calculating what shares would be based on our chosen parameters
	df["shares"] = np.exp(delta - alpha*df.p) / (1 + np.exp(delta - alpha*df.p))
	#
	#Making rows for j and t
	df["t"] = np.repeat(np.arange(T), J)
	df["j"] = np.tile(np.arange(J), T)
	#
	#Drawing demand from a poisson dist
	X_poisson = [] #generates J poison draws from P(lam*s) for each period
	arrivals = []
	for idx in np.arange(T):
		pk = df[df.t == idx].shares.values #These are the s values in this time period (there will be J of them)
		Q = []
		arrival = np.random.poisson(lambd) #only one num arrivals per period
		for i in range(pk.size): 
			Q.append(round(arrival*pk[i]))  # Generate Poisson-distributed samples
		X_poisson.append(Q)
		arrivals.append(arrival)
	#
	#
	X_poisson = np.array(X_poisson)
	arrivals = np.array(arrivals)
	dfS = pd.DataFrame(X_poisson) #only 90 time periods
	dfS["t"] = np.arange(T)
	dfL = pd.DataFrame(arrivals)
	dfL["t"] = np.arange(T)
	#
	# Reshaping so that demand (Q) is all in one column
	dfS = dfS.reset_index()
	dfTempS = dfS.drop("index", axis=1)
	dfTempS = dfTempS.drop("t", axis=1)
	dfS = dfS.melt(id_vars="t",value_vars=list(dfTempS.columns.values), value_name = 'Q')
	dfS.rename(columns={'variable': 'j'}, inplace=True)
	dfL.rename(columns={0: 'lam'}, inplace=True)
	#
	# Merging
	dfM = pd.merge(dfS, dfL, on='t', how='inner')
	#
	# Merging back into the original data
	df = df.merge(dfM, on=['t', 'j'])
	#
	df['lam_hat'] = df['lam'].mean() # here we take the cumulative mean
	return df

def maxLike(df):
	x_init = np.random.uniform(low=0, high=1)
	y_init = np.random.uniform(low=0, high=1)
	initial_guess = np.array([x_init, y_init])
	# Use the optimization function to find the solution, checking two different starting points
	result = minimize(fun = loglikelihood, x0 = initial_guess, args=(df,), method = 'L-BFGS-B', bounds = ((0.0, 1.0), (0.0,1.0)), options = {'gtol': 1e-15, 'ftol': 1e-15})
	return result

# MAIN FUNCTION
def main():
	# Initialize Parameters
	N = 1000
	J = 2
	T = 90
	delta = 0.5
	alpha = 0.03
	lambd = 30
	#
	# Simulate demand for N flights
	df = pd.DataFrame()
	for i in range(N):
		df_new = simFlightDemand(J,T,delta,alpha,lambd)
		df_new["sim"] = i
		df = pd.concat([df, df_new], ignore_index=True)
	#
	# Optimize 
	min_guess = np.array([1,1])
	# To account for local minima, we will do a loop and choose random initial values and take the smallest minimum we get
	for i in range(50):
		result = maxLike(df)
		if(objective(result.x, df) < objective(min_guess,df)):
			min_guess = result.x
	#
	return min_guess

