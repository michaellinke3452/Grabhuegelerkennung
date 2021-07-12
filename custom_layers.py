import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.backend import transpose, reverse 

# The Layers Slope, Reflectance, ShapeIndex, MeanCurvature, MinimalCurvature, 
# UnsphericityCurvature, Laplacian, SlopeTrainable, ReflectanceTrainable, 
# ShapeIndexTrainable, MeanCurvatureTrainable, MinimalCurvatureTrainable, 
# UnsphericityCurvatureTrainable, and LaplacianTrainable are based on the equations in 
# Florinsky, Igor V.: An illustrated introduction to general geomorphometry (2017).


class Rotate(layers.Layer): 
    def __init__(self, k=1): 
        super(Rotate, self).__init__() 
        self.k = k

    def build(self, input_dim): 
        self.input_dim = input_dim          
        super(Rotate, self).build(input_dim)    

    def rotation(self, inputs): 
        from tensorflow.keras.backend import reverse, permute_dimensions
        transposed = permute_dimensions(inputs, (0, 2, 1, 3)) 
        rev = reverse(transposed, axes=1) 
        return rev 

    def call(self, inputs):   
        for i in range(self.k): 
            inputs = self.rotation(inputs)    
        return inputs
    

class Gradient(layers.Layer): 
    def __init__(self, axis=0): 
        super(Gradient, self).__init__() 
        self.axis=axis 
    
    def build(self, input_dim): 
        super(Gradient, self).build(input_dim)  
        self.input_dim = input_dim  

    def get_gradient(self, matrix): 
        from tensorflow import roll 
        return roll(matrix, 1, axis=self.axis) - matrix   

    def call(self, inputs): 
        return tf.map_fn(self.get_gradient, inputs)
        

class Slope(layers.Layer): 
    def __init__(self): 
        super(Slope, self).__init__() 

    def build(self, input_dim): 
        super(Slope, self).build(input_dim) 
        self.input_dim = input_dim 
    
    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   

    def get_slope(self, inputs): 
        from tensorflow.math import atan, sqrt
        from utils import remove_nan 
        p = self.get_gradient(inputs, 1) 
        q = self.get_gradient(inputs, 2) 
        return remove_nan(atan(sqrt(p**2 + q**2)))
    
    def call(self, inputs): 
        return tf.map_fn(self.get_slope, inputs)



class Reflectance(layers.Layer): 
    def __init__(self, azimuth=5.497787, elevation=0.610865): 
        super(Reflectance, self).__init__()         
        #self.azimuth = tf.cast(azimuth, tf.float64) 
        #self.elevation = tf.cast(elevation, tf.float64) 
        self.azimuth = azimuth 
        self.elevation = elevation


    def build(self, input_dim): 
        super(Reflectance, self).build(input_dim) 
        self.input_dim = input_dim    

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   
    
    def cot(self, x): 
        from tensorflow.math import sin, cos 
        from utils import remove_nan 
        cot_ = cos(x) / sin(x)
        return  cot_

    def get_reflectance(self, inputs): 
        from tensorflow.math import sqrt, sin, cos 
        from utils import remove_nan       
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)        
        a = self.azimuth 
        e = self.elevation
        Z = 1 - p*sin(a) * self.cot(e) - q*cos(a) * self.cot(e) 
        N = (sqrt(1 + p**2 + q**2) * sqrt(1 + (sin(a) * self.cot(e)**2 + (cos(a) * self.cot(e))))) 
        r = Z / N        
        return r 

    def call(self, inputs):
        return tf.map_fn(self.get_reflectance, inputs) 
        


class ShapeIndex(layers.Layer): 
    def __init__(self): 
        super(ShapeIndex, self).__init__() 


    def build(self, input_dim): 
        super(ShapeIndex, self).build(input_dim) 
        self.input_dim = input_dim    


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   


    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)            


    def gaussian_curvature(self, p, q, r, s, t): 
        # (22)
        from utils import remove_nan
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return remove_nan(numerator / denominator) 


    def minimal_curvature(self, inputs): 
        # (19)
        from utils import remove_nan
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(H - tf.math.sqrt(H**2 - K)) 


    def maximal_curvature(self, inputs): 
        # (20)
        from utils import remove_nan
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(H + tf.math.sqrt(H**2 - K))


    def get_shape_index(self, inputs): 
        # (25)
        from tensorflow.math import sqrt
        from utils import remove_nan  
        from math import pi
        k_min = self.minimal_curvature(inputs) 
        k_max = self.maximal_curvature(inputs) 
        return remove_nan((2/pi) * tf.math.atan((k_max + k_min) / (k_max - k_min))) 
         

    def call(self, inputs):
        return tf.map_fn(self.get_shape_index, inputs) 
        





class MeanCurvature(layers.Layer): 
    def __init__(self): 
        super(MeanCurvature, self).__init__() 


    def build(self, input_dim): 
        super(MeanCurvature, self).build(input_dim) 
        self.input_dim = input_dim    


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   


    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)         


    def get_mean_curvature(self, inputs):         
        from tensorflow.math import sqrt
        from utils import remove_nan  
        from math import pi
        p, q, r, s, t = self.get_gradients(inputs)         
        return remove_nan(self.mean_curvature(p, q, r, s, t)) 
         

    def call(self, inputs):
        return tf.map_fn(self.get_mean_curvature, inputs) 




class MinimalCurvature(layers.Layer): 
    def __init__(self): 
        super(MinimalCurvature, self).__init__() 


    def build(self, input_dim): 
        super(MinimalCurvature, self).build(input_dim) 
        self.input_dim = input_dim    


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   


    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)            


    def gaussian_curvature(self, p, q, r, s, t): 
        # (22)
        from utils import remove_nan
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return remove_nan(numerator / denominator) 


    def minimal_curvature(self, inputs): 
        # (19)
        from utils import remove_nan
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(H - tf.math.sqrt(H**2 - K)) 


    def get_minimal_curvature(self, inputs):             
        from utils import remove_nan                  
        return remove_nan(self.minimal_curvature(inputs)) 
         

    def call(self, inputs):
        return tf.map_fn(self.get_minimal_curvature, inputs) 





class UnsphericityCurvature(layers.Layer): 
    def __init__(self): 
        super(UnsphericityCurvature, self).__init__() 


    def build(self, input_dim): 
        super(UnsphericityCurvature, self).build(input_dim) 
        self.input_dim = input_dim    


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   
        

    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)            


    def gaussian_curvature(self, p, q, r, s, t): 
        # (22)
        from utils import remove_nan
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return remove_nan(numerator / denominator) 


    def get_usc(self, inputs): 
        # (23)
        from tensorflow.math import sqrt
        from utils import remove_nan  
        from math import pi
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(sqrt(H**2 - K)) 
         

    def call(self, inputs):
        return tf.map_fn(self.get_usc, inputs) 
        



class Laplacian(layers.Layer): 
    def __init__(self): 
        super(Laplacian, self).__init__()         
    
    def build(self, input_dim): 
        super(Laplacian, self).build(input_dim) 
        self.input_dim = input_dim   

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        from utils import remove_nan 
        return remove_nan(roll(matrix, 1, axis) - matrix)          

    def get_laplacian(self, inputs):
        from utils import remove_nan 
        x = self.get_gradient(inputs, 1) 
        x = self.get_gradient(x, 1)
        y = self.get_gradient(inputs, 2) 
        y = self.get_gradient(y, 2)
        g = tf.clip_by_value(x + y, -1., 1.)
        g = g * tf.constant(0.5)
        #g = g + tf.constant(0.5)
        return  remove_nan(g)

    def call(self, inputs): 
        from utils import remove_nan
        inputs = remove_nan(inputs, 1.)
        return tf.map_fn(self.get_laplacian, inputs)


class Laplacian2(layers.Layer): 
    def __init__(self): 
        super(Laplacian2, self).__init__()         
    
    def build(self, input_dim): 
        super(Laplacian2, self).build(input_dim) 
        self.input_dim = input_dim   

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        from utils import remove_nan 
        return remove_nan(roll(matrix, 1, axis) - matrix)          

    def get_laplacian(self, inputs):
        from utils import remove_nan 
        x = self.get_gradient(inputs, 1) 
        x = self.get_gradient(x, 1)
        y = self.get_gradient(inputs, 2) 
        y = self.get_gradient(y, 2)        
        return  remove_nan(x + y)

    def call(self, inputs): 
        from utils import remove_nan
        inputs = remove_nan(inputs, 1.)
        return tf.map_fn(self.get_laplacian, inputs)



class SumOfGradients(layers.Layer): 
    def __init__(self): 
        super(SumOfGradients, self).__init__()         
    
    def build(self, input_dim): 
        super(SumOfGradients, self).build(input_dim) 
        self.input_dim = input_dim   

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        from utils import remove_nan 
        return remove_nan(roll(matrix, 1, axis) - matrix)          

    def get_sum_of_gradients(self, inputs):
        from utils import remove_nan 
        x = self.get_gradient(inputs, 1) 
        #x = self.get_gradient(x, 1)
        y = self.get_gradient(inputs, 2) 
        #y = self.get_gradient(y, 2)
        #g = tf.clip_by_value(x + y, -1., 1.)
        #g = g * tf.constant(0.5)
        #g = g + tf.constant(0.5)
        return  remove_nan(x + y)

    def call(self, inputs): 
        from utils import remove_nan
        inputs = remove_nan(inputs, 1.)
        return tf.map_fn(self.get_sum_of_gradients, inputs)





class Divide(layers.Layer): 
    def __init__(self, divisor=4.): 
        super(Divide, self).__init__() 
        self.divisor = divisor

    def build(self, input_dim): 
        super(Divide, self).build(input_dim) 
        self.input_dim = input_dim   


    def div(self, inputs): 
        #d = tf.ones(shape=self.input_dim)
        return inputs / self.divisor

    def call(self, inputs): 
        return self.div(inputs)




class Mean(layers.Layer): 
    def __init__(self): 
        super(Mean, self).__init__() 

    def build(self, input_dim): 
        super(Mean, self).build(input_dim) 
        self.input_dim = input_dim   


    def get_mean(self, inputs): 
        mean = tf.reshape(inputs[0], (64, 64) )
        print("Type Mean: ", type(mean))
        divisor = len(inputs)    
        print("Divisor: ", divisor)     
        for i in range(divisor):
            mean = tf.add(mean, tf.reshape(inputs[i], (64, 64)) ) 
        mean /= divisor 
        print("Type Mean: ", type(mean))
        return mean
    
    def call(self, inputs): 
        return self.get_mean(inputs)




class Threshold(layers.Layer): 
    def __init__(self): 
        super(Threshold, self).__init__() 
        

    def build(self, input_dim): 
        super(Threshold, self).build(input_dim) 
        #self.upper = self.add_weight(shape=(1,), name="upper", trainable=True) 
        self.input_dim = input_dim   
        self.upper = 1000.
        self.lower = self.add_weight(shape=(1,), name="lower", trainable=True)

    def clip(self, inputs): 
        inputs = tf.clip_by_value(inputs, self.lower, self.upper) 
        inputs = tf.clip_by_value(inputs - self.lower, 0., self.upper)
        return inputs

    def call(self, inputs): 
        return self.clip(inputs)

 













class SlopeTrainable(layers.Layer): 
    def __init__(self): 
        super(SlopeTrainable, self).__init__() 

    def build(self, input_dim): 
        super(SlopeTrainable, self).build(input_dim) 
        self.input_dim = input_dim 
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)
    
    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   

    def get_slope(self, inputs): 
        from tensorflow.math import atan, sqrt
        from utils import remove_nan 
        p = self.get_gradient(inputs, 1) 
        q = self.get_gradient(inputs, 2) 
        return remove_nan(atan(sqrt(p**2 + q**2)))
    
    def call(self, inputs): 
        return self.factor * tf.map_fn(self.get_slope, inputs)



class ReflectanceTrainable(layers.Layer): 
    def __init__(self, azimuth=5.497787, elevation=0.610865): 
        super(ReflectanceTrainable, self).__init__()         
        #self.azimuth = tf.cast(azimuth, tf.float64) 
        #self.elevation = tf.cast(elevation, tf.float64) 
        self.azimuth = azimuth 
        self.elevation = elevation


    def build(self, input_dim): 
        super(ReflectanceTrainable, self).build(input_dim) 
        self.input_dim = input_dim  
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)  

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   
    
    def cot(self, x): 
        from tensorflow.math import sin, cos 
        from utils import remove_nan 
        cot_ = cos(x) / sin(x)
        return  cot_

    def get_reflectance(self, inputs): 
        from tensorflow.math import sqrt, sin, cos 
        from utils import remove_nan       
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)        
        a = self.azimuth 
        e = self.elevation
        Z = 1 - p*sin(a) * self.cot(e) - q*cos(a) * self.cot(e) 
        N = (sqrt(1 + p**2 + q**2) * sqrt(1 + (sin(a) * self.cot(e)**2 + (cos(a) * self.cot(e))))) 
        r = Z / N        
        return r 

    def call(self, inputs):
        return self.factor *  tf.map_fn(self.get_reflectance, inputs) 
        


class ShapeIndexTrainable(layers.Layer): 
    def __init__(self): 
        super(ShapeIndexTrainable, self).__init__() 


    def build(self, input_dim): 
        super(ShapeIndexTrainable, self).build(input_dim) 
        self.input_dim = input_dim 
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)   


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   


    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)            


    def gaussian_curvature(self, p, q, r, s, t): 
        # (22)
        from utils import remove_nan
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return remove_nan(numerator / denominator) 


    def minimal_curvature(self, inputs): 
        # (19)
        from utils import remove_nan
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(H - tf.math.sqrt(H**2 - K)) 


    def maximal_curvature(self, inputs): 
        # (20)
        from utils import remove_nan
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(H + tf.math.sqrt(H**2 - K))


    def get_shape_index(self, inputs): 
        # (25)
        from tensorflow.math import sqrt
        from utils import remove_nan  
        from math import pi
        k_min = self.minimal_curvature(inputs) 
        k_max = self.maximal_curvature(inputs) 
        return remove_nan((2/pi) * tf.math.atan((k_max + k_min) / (k_max - k_min))) 
         

    def call(self, inputs):
        return self.factor * tf.map_fn(self.get_shape_index, inputs) 
        





class MeanCurvatureTrainable(layers.Layer): 
    def __init__(self): 
        super(MeanCurvatureTrainable, self).__init__() 


    def build(self, input_dim): 
        super(MeanCurvatureTrainable, self).build(input_dim) 
        self.input_dim = input_dim 
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)   


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   


    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)         


    def get_mean_curvature(self, inputs): 
        # (25)
        from tensorflow.math import sqrt
        from utils import remove_nan  
        from math import pi
        p, q, r, s, t = self.get_gradients(inputs)         
        return remove_nan(self.mean_curvature(p, q, r, s, t)) 
         

    def call(self, inputs):
        return self.factor *  tf.map_fn(self.get_mean_curvature, inputs) 




class MinimalCurvatureTrainable(layers.Layer): 
    def __init__(self): 
        super(MinimalCurvatureTrainable, self).__init__() 


    def build(self, input_dim): 
        super(MinimalCurvatureTrainable, self).build(input_dim) 
        self.input_dim = input_dim   
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True) 


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   


    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)            


    def gaussian_curvature(self, p, q, r, s, t): 
        # (22)
        from utils import remove_nan
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return remove_nan(numerator / denominator) 


    def minimal_curvature(self, inputs): 
        # (19)
        from utils import remove_nan
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(H - tf.math.sqrt(H**2 - K)) 


    def get_minimal_curvature(self, inputs):             
        from utils import remove_nan                  
        return remove_nan(self.minimal_curvature(inputs)) 
         

    def call(self, inputs):
        return self.factor *  tf.map_fn(self.get_minimal_curvature, inputs) 





class UnsphericityCurvatureTrainable(layers.Layer): 
    def __init__(self): 
        super(UnsphericityCurvatureTrainable, self).__init__() 


    def build(self, input_dim): 
        super(UnsphericityCurvatureTrainable, self).build(input_dim) 
        self.input_dim = input_dim    
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)


    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        return roll(matrix, 1, axis) - matrix   


    def get_gradients(self, inputs):
        p = self.get_gradient(inputs, 1)  
        q = self.get_gradient(inputs, 2)   
        r = self.get_gradient(p, 1)
        s = self.get_gradient(p, 2)
        t = self.get_gradient(q, 2)
        return p, q, r, s, t   
        

    def mean_curvature(self, p, q, r, s, t): 
        # (21)
        from utils import remove_nan
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * tf.math.sqrt((1 + p**2 + q**2)**3)     
        return remove_nan(numerator / denominator)            


    def gaussian_curvature(self, p, q, r, s, t): 
        # (22)
        from utils import remove_nan
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return remove_nan(numerator / denominator) 


    def get_usc(self, inputs): 
        # (12)
        from tensorflow.math import sqrt
        from utils import remove_nan  
        from math import pi
        p, q, r, s, t = self.get_gradients(inputs) 
        H = self.mean_curvature(p, q, r, s, t)
        K = self.gaussian_curvature(p, q, r, s, t)
        return remove_nan(sqrt(H**2 - K)) 
         

    def call(self, inputs):
        return self.factor *  tf.map_fn(self.get_usc, inputs) 
        



class LaplacianTrainable(layers.Layer): 
    def __init__(self): 
        super(LaplacianTrainable, self).__init__()         
    
    def build(self, input_dim): 
        super(LaplacianTrainable, self).build(input_dim) 
        self.input_dim = input_dim   
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        from utils import remove_nan 
        return remove_nan(roll(matrix, 1, axis) - matrix)          

    def get_laplacian(self, inputs):
        from utils import remove_nan 
        x = self.get_gradient(inputs, 1) 
        x = self.get_gradient(x, 1)
        y = self.get_gradient(inputs, 2) 
        y = self.get_gradient(y, 2)
        g = tf.clip_by_value(x + y, -1., 1.)
        g = g * tf.constant(0.5)
        #g = g + tf.constant(0.5)
        return  remove_nan(g)

    def call(self, inputs): 
        from utils import remove_nan
        inputs = remove_nan(inputs, 1.)
        return self.factor *  tf.map_fn(self.get_laplacian, inputs)





class SumOfGradientsTrainable(layers.Layer): 
    def __init__(self): 
        super(SumOfGradientsTrainable, self).__init__()         
    
    def build(self, input_dim): 
        super(SumOfGradientsTrainable, self).build(input_dim) 
        self.input_dim = input_dim   
        self.factor = self.add_weight(shape=(1,), name="factor", trainable=True)

    def get_gradient(self, matrix, axis): 
        from tensorflow import roll 
        from utils import remove_nan 
        return remove_nan(roll(matrix, 1, axis) - matrix)          

    def get_sum_of_gradients(self, inputs):
        from utils import remove_nan 
        x = self.get_gradient(inputs, 1) 
        #x = self.get_gradient(x, 1)
        y = self.get_gradient(inputs, 2) 
        #y = self.get_gradient(y, 2)
        #g = tf.clip_by_value(x + y, -1., 1.)
        #g = g * tf.constant(0.5)
        #g = g + tf.constant(0.5)
        return  remove_nan(x + y)

    def call(self, inputs): 
        from utils import remove_nan
        inputs = remove_nan(inputs, 1.)
        return self.factor *  tf.map_fn(self.get_sum_of_gradients, inputs)

