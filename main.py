import math
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import util


# Part 1 Collecting historical data
print("**********Part 1**********")
stock_prices = pd.read_excel('./assignment 1 stock data.xlsx', sheet_name=1)
stock_num = 10  # k = 10
days = len(stock_prices.index)  # 494
S = []
for i in range(0, days):
    tmp = []
    for j in range(0, stock_num):
        tmp.append(stock_prices.values[i, j + 1])
    S.append(tmp)

R = []
for i in range(1, days):
    tmp = []
    for j in range(0, stock_num):
        tmp.append(S[i][j] / S[i - 1][j] - 1)
    R.append(tmp)

R_mean = []
for i in range(0, stock_num):
    tmp = 0
    for j in range(0, days - 1):
        tmp += R[j][i]
    tmp /= (days - 1)
    R_mean.append(tmp)

standard_deviation = []
for i in range(0, stock_num):
    tmp = 0
    for j in range(0, days - 1):
        tmp = tmp + (R[j][i] - R_mean[i]) * (R[j][i] - R_mean[i])
    tmp = math.sqrt(252 / (days - 2) * tmp)
    standard_deviation.append(tmp)
print("Annualized standard deviation of daily return:")
print(standard_deviation)

covariance = []
for i in range(0, stock_num):
    row = []
    for j in range(0, stock_num):
        tmp = 0
        for n in range(0, days - 1):
            tmp += (R[n][i] - R_mean[i]) * (R[n][j] - R_mean[j])
        tmp = 252 * tmp / (days - 2)
        row.append(tmp)
    covariance.append(row)
print('Annualized covariance between each pair of stocks:')
print(covariance)

correlation = []
for i in range(0, stock_num):
    tmp = []
    for j in range(0, stock_num):
        tmp.append(covariance[i][j] / (standard_deviation[i] * standard_deviation[j]))
    correlation.append(tmp)
print('Correlation coefficient between each pair of stocks:')
print(correlation)
print()

# Part 2 i.a
print("**********Part 2 i.a**********")
returns = util.getReturns()[0]
update_returns = util.getReturns()[1]
weight = np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
print('When all the stocks have an equal weight of 0.1:')
print('expected_return = ' + str(util.statistics(weight, returns, covariance)[0]))
print('standard deviation = ' + str(util.statistics(weight, returns, covariance)[1]))
print()

# Part 2 i.b
print("**********Part 2 i.b**********")
# 设计初始值，weight均为0.1
w0 = np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
bounds = tuple((0, 1) for x in range(stock_num))
optv = sco.minimize(lambda x: util.statistics(x, returns, covariance)[1], w0, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
expected_weight = optv['x']
print('expected weights of the stocks = ', end='')
print(expected_weight)
print('standard deviation = ' + str(optv['fun']))
print('expected return of the portfolio = ' + str(util.statistics(expected_weight, returns, covariance)[0]))
print()

# Part 2 i.c
print("**********Part 2 i.c**********")
target_returns = np.linspace(0.0, 0.3, 100)
target_variance = []
for target in target_returns:
    cons = ({'type': 'eq', 'fun': lambda x: util.statistics(x, returns, covariance)[0] - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(lambda x: util.statistics(x, returns, covariance)[1], w0, method='SLSQP', bounds=bounds, constraints=cons)
    target_variance.append(res['fun'])
target_variance = np.array(target_variance[21:72])
target_returns = target_returns[21:72]
plt.figure(figsize=(8, 4))
plt.plot(target_variance, target_returns, marker='.')
plt.plot(util.statistics(expected_weight, returns, covariance)[1], util.statistics(expected_weight, returns, covariance)[0], 'y*', markersize=15.0)
plt.grid(True)
plt.xlabel('standard deviation')
plt.ylabel('expected return')
plt.savefig("efficient frontier.png")
print('The figure is saved as "efficient frontier.png", you can find it in the directory.')
print()

# Part 2 i.d
print("**********Part 2 i.d**********")
# 还是用拉格朗日乘数法算weights
optv = sco.minimize(lambda x: -util.statistics(x, returns, covariance)[2],
                    w0, method='SLSQP', bounds=bounds,
                    constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
print('the optimal portfolio to be combined with the risk free asset')
std_deviation = util.statistics(optv.x, returns, covariance)[1]
exp_return = 0.043 + (-optv.fun) * std_deviation
print('standard deviation = ' + str(std_deviation))
print('expected return = ' + str(exp_return))
print('expected weights of the stocks = ', end='')
print(optv.x)
print()


# Part 2 i.e
print("**********Part 2 i.e**********")
optv = sco.minimize(lambda x: -util.statistics(x, update_returns, covariance)[2],
                    w0, method='SLSQP', bounds=bounds,
                    constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
print('expected weights of the stocks = ', end='')
print(optv.x)
print()


# Part 2 ii.a
print("**********Part 2 ii.a**********")
optv = sco.minimize(lambda x: util.risk_parity(x, covariance),
                    w0, method='SLSQP', bounds=bounds,
                    constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
print('total risk contribution = ' + str(optv.fun))
print('weights of the stocks = ', end='')
print(optv.x)
print()


# Part 2 ii.b
print("**********Part 2 ii.b**********")



