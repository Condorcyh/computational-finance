import pandas as pd
import numpy as np


risk_free_return = 0.043


def getReturns():
    consensus_expected_return = pd.read_excel('./assignment 1 stock data.xlsx', sheet_name=0)
    returns = []
    update_returns = []
    for i in range(0, 10):
        returns.append(consensus_expected_return.values[i][2])
        update_returns.append(consensus_expected_return.values[i][3])
    return np.array([returns, update_returns])


def statistics(w, r, covariance):
    weights = np.array(w)
    portfolio_returns = np.sum(r * weights)
    portfolio_variance = np.sqrt(np.dot(weights, np.dot(covariance, weights.T)))
    return np.array([portfolio_returns, portfolio_variance, (portfolio_returns - risk_free_return) / portfolio_variance])


def risk_parity(w, cov):
    weights = np.array(w)
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    MR = np.dot(cov, weights) / sigma
    TR = weights * MR
    delta_TR = 0
    for i in range(0, 10):
        for j in range(0, 10):
            delta_TR = delta_TR + (TR[i] - TR[j]) * (TR[i] - TR[j])

    return delta_TR

