from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

xs = np.array([1,3,6,9], dtype=np.float64)
ys = np.array([4,7,10,11], dtype=np.float64)

def best_fit_line(x,y):
    m = ((mean(x)*(mean(y))) - (mean(x*y)))/((mean(x)**2) - mean(x**2))
    b = mean(y) - m*(mean(x))
    return m,b

def square_error(y1,y2):
    "Calculating square errors since the errors can hold a positive as well as a negative value"
    return (sum((y1-y2)**2))

def coeff_of_determination(y_orig, y_line):
    "Returns about 92% coeff of determination"
    y_mean_line = [mean(y_orig) for y in y_orig]
    square_error_y_mean = square_error(y_orig, y_mean_line)
    square_error_reg = square_error(y_orig, y_line)
    r2 = 1 - (square_error_reg/square_error_y_mean)
    return r2

m,b = best_fit_line(xs, ys)
regression_line = [(m*x)+b for x in xs]
plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()
print(coeff_of_determination(ys, regression_line))
