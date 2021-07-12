import tensorflow as tf
import numpy as np



def reflectance(p, q): 
    from tf.math import sin, cos, cot, sqrt
    a = np.deg2rad(315) 
    e = np.deg2rad(35)      
    R = 1 - p*sin(a) * cot(e) - q*cos(a) * cot(e) 
    R /= (np.sqrt(1 + p**2 + q**2) * np.sqrt(1 + (sin(a) * cot(e)**2 + (cos(a) * cot(e))))) 
    return np.nan_to_num(R)  