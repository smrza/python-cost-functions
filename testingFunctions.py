import numpy as np
import matplotlib.pyplot as plt
import os

##########################
# functions
##########################

# 1
def ackley( x, a=20, b=0.2, c=2*np.pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( np.cos( c * x ))
    return -a*np.exp( -b*np.sqrt( s1 / n )) - np.exp( s2 / n ) + a + np.exp(1)

def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = np.prod( np.cos( x / np.sqrt(j) ))
    return s/fr - p + 1

functions = [ackley, griewank]

##########################
# charting
##########################

def chart_3d_function(function, x_range=(-100, 100), y_range=(-100, 100)):
    function_name = function.__name__
    dir = f'charts/{function_name}'
    os.makedirs(dir, exist_ok=True)

    x = np.linspace(x_range[0], x_range[1])
    y = np.linspace(y_range[0], y_range[1])
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j] = function([X[i, j], Y[i, j]])

    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')
    axes.plot_surface(X, Y, Z, cmap='plasma')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    plt.title('3D Chart: ' + function_name)
    path = os.path.join(dir, f'3D_{function_name}.png')
    plt.savefig(path)
    plt.clf()


def chart_2d_function(function, x_range=(-100, 100)):
    function_name = function.__name__
    dir = f'charts/{function_name}'
    os.makedirs(dir, exist_ok=True)

    x = np.linspace(x_range[0], x_range[1])
    y = [function([x_val]) for x_val in x]

    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Chart: ' + function_name)
    path = os.path.join(dir, f'2D_{function_name}.png')
    plt.savefig(path)
    plt.clf()


# main

for function in functions:
    chart_3d_function(function)
    chart_2d_function(function)
