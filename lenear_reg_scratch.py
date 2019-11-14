import numpy as np
import matplotlib.pyplot as plt


def compute_cost(X,y,theta):
    m = len(y)
    ji = list(range(1,m))
    sum1 = 0
    for i in range(m):
        sum1 = sum1+ ((theta[0]+X[i]*theta[1])-y[i])**2

    J = (1/(2*m))*sum1
    print(J)
    return J

def gradient_descent(X,y,theta,alpha,interation):
    m = len(y)
    J_history = np.zeros([interation])

    for i in range(interation):
        t1=0
        t2=0
        for i in range(m):
            t1=(theta[0]+theta[1]*X[i])-y[i]
            t2 = t1*X[i]

        theta[0] = theta[0] -((alpha/m)*t1)
        theta[1] = theta[1] - ((alpha / m) * t2)
        J_history[i] =compute_cost(X,y,theta)

    return theta


        


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()
    print("show")


def main():
    # observations
    theta = np.zeros([2])
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    #theta = np.zeros([2])
    #compute_cost(x,y,theta)\
    res =gradient_descent(x,y,theta,0.01,1500)

    plot_regression_line(x,y,res)

    # estimating coefficients


if __name__ == "__main__":
    main()