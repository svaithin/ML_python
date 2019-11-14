import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()
    print("show")

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X = np.array(x).reshape(len(x), 1)
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

reg = LinearRegression().fit(X, y)
print("coeff",reg.coef_)
print(reg.score(X, y))
print(reg.intercept_)
plot_regression_line(X,y,reg.coef_)