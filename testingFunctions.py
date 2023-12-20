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

# 2
def griewank( x, fr=4000 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    s = sum( x**2 )
    p = np.prod( np.cos( x / np.sqrt(j) ))
    return s/fr - p + 1

# 3
def levy( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    z = 1 + (x - 1) / 4
    return (np.sin( np.pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * np.sin( np.pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + np.sin( 2 * np.pi * z[-1] )**2 ))

# 4
michalewicz_m = .5  # orig 10: ^20 => underflow
def michalewicz( x ):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( np.sin(x) * np.sin( j * x**2 / np.pi ) ** (2 * michalewicz_m) )

# 5
def dixonprice( x ):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2

# 6
def perm( x, b=.5 ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 1., n+1 )
    xbyj = np.fabs(x) / j
    return np.mean([ np.mean( (j**k + b) * (xbyj ** k - 1) ) **2
            for k in j/n ])
    # original overflows at n=100 --
    # return sum([ sum( (j**k + b) * ((x / j) ** k - 1) ) **2
    #       for k in j ])

# 7
def powersum( x, b=[8,18,44,114] ):  # power.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
    return s

# 8
def rastrigin( x ):  # rast.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 10*n + sum( x**2 - 10 * np.cos( 2 * np.pi * x ))

# 9
def rosenbrock( x ):  # rosen.m
    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
        # a sum of squares, so LevMar (scipy.optimize.leastsq) is pretty good
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (sum( (1 - x0) **2 )
        + 100 * sum( (x1 - x0**2) **2 ))

# 10
def schwefel( x ):  # schw.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829*n - sum( x * np.sin( np.sqrt( abs( x ))))

# 11
def trid( x ):
    x = np.asarray_chkfinite(x)
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

# 12
def nesterov( x ):
    """ Nesterov's nonsmooth Chebyshev-Rosenbrock function, Overton 2011 variant 2 """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return abs( 1 - x[0] ) / 4 \
        + sum( abs( x1 - 2*abs(x0) + 1 ))

# 13
def alpine_n1(x):
    x = np.asarray_chkfinite(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

# 14
def qing(x):
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        result += (x[i] ** 2 - i - 1) ** 2
    return result

# 15
def salomon(x):
    x = np.asarray_chkfinite(x)
    return 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x ** 2))) + 0.1 * np.sqrt(np.sum(x ** 2))

# 16
def styblinski(x):
    x = np.asarray_chkfinite(x)
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)

# 17
def happy_cat(x, alpha=1.0/8):
    """
    Class: multimodal, non-convex, differentiable, non-separable, parametric
    Global: one global minimum fx = 0, at [-1, ..., -1]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/happycatfcn.html

    @param solution: A numpy array with x_i in [-2, 2]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    return ((np.sum(x**2) - len(x))**2)**alpha + (0.5*np.sum(x**2)+np.sum(x))/len(x) + 0.5

# 18
def quartic(x):
    """
    Class: multimodal, non-convex, differentiable, separable, continuous, random
    Global: one global minimum fx = 0 + random, at (0, ...,0)
    Link: http://benchmarkfcns.xyz/benchmarkfcns/quarticfcn.html

    @param solution: A numpy array with x_i in [-1.28, 1.28]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        result+= (i+1)*x[i]**4
    return result+np.random.uniform(0, 1)

# 19
def shubert_3(x):
    """
    Class: multi-modal, non-convex, differentiable, separable, continuous
    Global: one global minimum fx = -29.6733337
    Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert3fcn.html

    @param solution: A numpy array with x_i in [-10, 10]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        for j in range(1, 6):
            result+= j*np.sin((j+1)*x[i] + j)
    return result

# 20
def shubert_4(x):
    """
    Class: multi-modal, non-convex, differentiable, separable, continuous
    Global: one global minimum fx = -25.740858
    Link: http://benchmarkfcns.xyz/benchmarkfcns/shubert4fcn.html

    @param solution: A numpy array with x_i in [-10, 10]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    result = 0
    for i in range(0, d):
        for j in range(1, 6):
            result += j * np.cos((j + 1) * x[i] + j)
    return result

# 21
def shubert(x):
    """
    Class: multi-modal, non-convex, differentiable, non-separable, continuous
    Global: one global minimum fx = 0, at [0, ..., 0]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/shubertfcn.html

    @param solution: A numpy array with x_i in [-100, 100]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    prod = 1.0
    for i in range(0, d):
        result = 0
        for j in range(1, 6):
            result += np.cos((j + 1) * x[i] + j)
        prod *= result
    return prod

# 22
def ackley_n4(x):
    """
    Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
    Global: on 2-d space, 1 global min fx = -4.590101633799122, at [−1.51, −0.755]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/ackleyn4fcn.html

    @param solution: A numpy array include 2 items like: [-35, 35, -35, ...]
    """
    x = np.asarray_chkfinite(x)
    d = len(x)
    score = 0.0
    for i in range(0, d-1):
        score += ( np.exp(-0.2*np.sqrt(x[i]**2 + x[i+1]**2)) + 3*(np.cos(2*x[i]) + np.sin(2*x[i+1])) )
    return score

# 23
def alpine_n2(x):
    """
    Class: multimodal, non-convex, differentiable, non-separable, n-dimensional space.
    Global: one global minimum fx = 2.808^n, at [7.917, ..., 7.917]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/alpinen2fcn.html

    @param solution: A numpy array like: [1, 2, 10, 4, ...]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    x_abs = np.abs(x)
    return np.prod(np.sqrt(x_abs)*np.sin(x))

# 24
def xin_she_yang_n2(x):
    """
    Class: multi-modal, non-convex, non-differentiable, non-separable
    Global: one global minimum fx = 0, at [0, ..., 0]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn2fcn.html

    @param solution: A numpy array with x_i in [-2pi, 2pi]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    return np.sum(np.abs(x))*np.exp(-np.sum(np.sin(x**2)))

# 25
def xin_she_yang_n4(x):
    """
    Class: multi-modal, non-convex, non-differentiable, non-separable
    Global: one global minimum fx = -1, at [0, ..., 0]
    Link: http://benchmarkfcns.xyz/benchmarkfcns/xinsheyangn4fcn.html

    @param solution: A numpy array with x_i in [-10, 10]
    @return: fx
    """
    x = np.asarray_chkfinite(x)
    t1 = np.sum(np.sin(x)**2)
    t2 = -np.exp(-np.sum(x**2))
    t3 = -np.exp(np.sum(np.sin(np.sqrt(np.abs(x)))**2))
    return (t1 + t2) * t3
    
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
    
functions = [ackley, griewank, levy, michalewicz, dixonprice, perm, powersum, rastrigin, rosenbrock, schwefel, trid, nesterov, 
             alpine_n1, qing, salomon, styblinski, happy_cat, quartic, shubert_3, shubert_4, shubert, ackley_n4, alpine_n2,
             xin_she_yang_n2, xin_she_yang_n4]

for function in functions:
    chart_3d_function(function)
    chart_2d_function(function)
