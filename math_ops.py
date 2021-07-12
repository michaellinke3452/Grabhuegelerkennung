import numpy as np 



def emp_corr(x, y): 
    # empirical correlation coefficient.
    #x and y must have the same shape.
    x_ = x.flatten()
    y_ = y.flatten()
    #print(np.mean(x_ - y_)) 
    #print()
    mean_x = np.mean(x) 
    mean_y = np.mean(y) 
    a = (x_ - mean_x) * (y_ - mean_y)
    sum_a = np.sum(a) 
    b = (x_ - mean_x)**2 * (y_ - mean_y)**2 
    b = np.sqrt(b) 
    sum_b = np.sum(b) 
    #print(sum(a) / sum(b))
    return sum(a) / sum(b)