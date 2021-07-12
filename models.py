import tensorflow as tf
from custom_metrics import *


def create_conv_segmentation_model(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model
    optimizer = "adam"
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(16, (3,3), activation="relu", padding="same")(inputs) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(8, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(8, (3,3), activation= "relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 

    x = Conv2D(16, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 

    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    return model

def create_conv_segmentation_model6(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D
    from tensorflow.keras.models import Model
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    #optimizer = "adam"
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs) 
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    
    x = Conv2D(32, (3,3), activation= "relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 

    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    return model



def create_conv_segmentation_model7(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs) 
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    
    x = Conv2D(64, (3,3), activation= "relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 
    #x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 

    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss)

    return model



def create_conv_segmentation_model8(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs) 
    y = Dropout(0.2)(x)
    y = MaxPooling2D((2,2), padding="same")(y) 

    y = Conv2D(64, (3,3), activation="relu", padding="same")(y) 
    z = MaxPooling2D((2,2), padding="same")(y) 

    z = Conv2D(128, (3,3), activation="relu", padding="same")(z)
    
    z = Conv2D(64, (3,3), activation= "relu", padding="same")(z) 
    z = UpSampling2D((2,2))(z) 
    #x = BatchNormalization()(x)
    v = Concatenate()([z, y])
    v = Conv2D(32, (3,3), activation="relu", padding="same")(v) 
    v = UpSampling2D((2,2))(v) 
    w = Concatenate()([v, x])
    w = Conv2D(32, (3,3), activation="relu", padding="same")(w)

    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(w) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss)

    return model





def create_multiscale_segmentation_model(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    #he = "glorot_uniform"
    inputs = Input(shape=(width, height, 1))     
    x = Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=he)(inputs) 
    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_concat = Concatenate()([x, x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(40, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model


def create_multiscale_segmentation_model3(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    #he = "glorot_uniform"
    
    inputs = Input(shape=(width, height, 1))     
    x = Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=he)(inputs) 
    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_concat = Concatenate()([x, x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(128, (1,1), activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x = Conv2D(40, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    x1 = MaxPooling2D((2,2), padding="same")(x) 

    x1 = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x1) 
    x2 = MaxPooling2D((2,2), padding="same")(x1) 

    x2 = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x2)    
    
    x3 = UpSampling2D((2,2))(x2)     
    x3 = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x3) 
    x3_concat = Concatenate()([x1, x3]) 
    x3_concat = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x3_concat)

    x4 = UpSampling2D((2,2))(x3_concat) 
    #x5 = Concatenate()([x_concat, x4])
    x5 = Concatenate()([x, x4])
    x5 = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x5)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x5) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model



def create_multifilter_segmentation_model(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    from custom_layers import Reflectance
    from custom_layers import Laplacian, ShapeIndex, Slope   

    optimizer = "adam"    
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    
    inputs = Input(shape=(width, height, 1))  
    slope = Slope()(inputs) 
    reflectance = Reflectance()(inputs) 
    shape_index = ShapeIndex()(inputs) 
    laplacian = Laplacian()(inputs) 
    x_concat = Concatenate()([slope, reflectance, shape_index, laplacian]) 



    x = Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer=he)(x_concat) 
    #x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(inputs)
    #x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(inputs)
    #x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(inputs)
    #x_concat = Concatenate()([x, x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (1,1), activation="relu", kernel_initializer=he, padding="same")(x)
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    x1 = MaxPooling2D((2,2), padding="same")(x) 

    x1 = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x1) 
    x2 = MaxPooling2D((2,2), padding="same")(x1) 

    x2 = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x2)    
    
    x3 = UpSampling2D((2,2))(x2)     
    x3 = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x3) 
    x3_concat = Concatenate()([x1, x3]) 
    x3_concat = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x3_concat)

    x4 = UpSampling2D((2,2))(x3_concat) 
    #x5 = Concatenate()([x_concat, x4])
    x5 = Concatenate()([x, x4])
    x5 = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x5)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x5) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model


def create_multifilter_segmentation_model2(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    from custom_layers import Reflectance
    from custom_layers import Laplacian, ShapeIndex, Slope, Rotate       

    optimizer = "adam"    
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    
    inputs = Input(shape=(width, height, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])

    si1 = ShapeIndex()(inputs) 
    si2 = ShapeIndex()(x_rotated1) 
    si3 = ShapeIndex()(x_rotated2) 
    si4 = ShapeIndex()(x_rotated3)
    si2 = Rotate(k=3)(si2)       
    si3 = Rotate(k=2)(si3)   
    si4 = Rotate(k=1)(si4) 
    shape_index = Concatenate()([si1, si2, si3, si4])
    
    laplacian = Laplacian()(inputs) 
    x_concat = Concatenate()([slope, reflectance, shape_index, laplacian]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=he)(x_concat) 
    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_concat = Concatenate()([x, x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (1,1), activation="relu", kernel_initializer=he, padding="same")(x)
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    x1 = MaxPooling2D((2,2), padding="same")(x) 

    x1 = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x1) 
    x2 = MaxPooling2D((2,2), padding="same")(x1) 

    x2 = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x2)    
    
    x3 = UpSampling2D((2,2))(x2)     
    x3 = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x3) 
    x3_concat = Concatenate()([x1, x3]) 
    x3_concat = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x3_concat)

    x4 = UpSampling2D((2,2))(x3_concat) 
    #x5 = Concatenate()([x_concat, x4])
    x5 = Concatenate()([x, x4])
    x5 = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x5)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x5) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model


def create_multifilter_segmentation_model3(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    from custom_layers import Reflectance
    from custom_layers import Laplacian, ShapeIndex, Slope, Rotate       

    optimizer = "adam"    
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    
    inputs = Input(shape=(width, height, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])

    si1 = ShapeIndex()(inputs) 
    si2 = ShapeIndex()(x_rotated1) 
    si3 = ShapeIndex()(x_rotated2) 
    si4 = ShapeIndex()(x_rotated3)
    si2 = Rotate(k=3)(si2)       
    si3 = Rotate(k=2)(si3)   
    si4 = Rotate(k=1)(si4) 
    shape_index = Concatenate()([si1, si2, si3, si4])
    
    laplacian = Laplacian()(inputs) 
    x_concat = Concatenate()([slope, reflectance, shape_index, laplacian]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=he)(x_concat) 
    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_concat = Concatenate()([x, x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (1,1), activation="relu", kernel_initializer=he, padding="same")(x)
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    x1 = MaxPooling2D((2,2), padding="same")(x) 

    x1 = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x1) 
    x2 = MaxPooling2D((2,2), padding="same")(x1) 

    x2 = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x2)    
    
    x3 = UpSampling2D((2,2))(x2)     
    x3 = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x3) 
    x3_concat = Concatenate()([x1, x3]) 
    x3_concat = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x3_concat)

    x4 = UpSampling2D((2,2))(x3_concat) 
    #x5 = Concatenate()([x_concat, x4])
    x5 = Concatenate()([x, x4])
    x5 = Conv2D(32, (1,1), activation="relu", kernel_initializer=he, padding="same")(x5)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x5) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model




def create_multifilter_segmentation_model4(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])

    si1 = ShapeIndex()(inputs) 
    si2 = ShapeIndex()(x_rotated1) 
    si3 = ShapeIndex()(x_rotated2) 
    si4 = ShapeIndex()(x_rotated3)
    si2 = Rotate(k=3)(si2)       
    si3 = Rotate(k=2)(si3)   
    si4 = Rotate(k=1)(si4) 
    shape_index = Concatenate()([si1, si2, si3, si4])
    
    laplacian = Laplacian()(inputs) 
    x_concat = Concatenate()([slope, reflectance, shape_index, laplacian]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)
    #y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)
    #y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    #x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    #x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    #x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    return model




def create_multifilter_segmentation_model5(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(None, None, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    shape_index = ShapeIndex()(inputs)
    
    laplacian = Laplacian()(inputs) 
    x_concat = Concatenate()([slope, reflectance, shape_index, laplacian]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)    
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)  

    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)   

    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    model.summary()
    return model


def create_multifilter_segmentation_model5a(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate
    optimizer = tf.keras.optimizers.Adam() #lr=0.0001
    loss = f1_loss

    inputs = Input(shape=(None, None, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    shape_index = ShapeIndex()(inputs)
    
    laplacian = Laplacian()(inputs) 
    x_concat = Concatenate()([slope, reflectance, shape_index, laplacian]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)    
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)  

    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)   

    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    model.summary()
    return model


def create_multifilter_segmentation_model6(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(None, None, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 

    x_concat = Concatenate()([slope, reflectance, usc, shape_index, laplacian]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)    
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)  

    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)   

    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    model.summary()
    return model



def create_multifilter_segmentation_model7(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(None, None, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    #shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 

    x_concat = Concatenate()([slope, reflectance, usc, laplacian]) #, shape_index

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)    
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)  

    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)   

    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    model.summary()
    return model


def create_multifilter_segmentation_model8(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(None, None, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    #laplacian = Laplacian()(inputs) 

    x_concat = Concatenate()([slope, reflectance, usc, shape_index]) #, laplacian

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)    
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)  

    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)   

    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    model.summary()
    return model



def create_multifilter_segmentation_model9(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization    
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(None, None, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    #laplacian = Laplacian()(inputs) 

    slope = Conv2D(16, (3,3), activation="relu", padding="same")(slope) 
    reflectance = Conv2D(16, (3,3), activation="relu", padding="same")(reflectance)
    usc = Conv2D(16, (3,3), activation="relu", padding="same")(usc)
    shape_index = Conv2D(16, (3,3), activation="relu", padding="same")(shape_index)



    x_c = Concatenate()([slope, reflectance, usc, shape_index]) #, laplacian

    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    #x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x)    
    #x = Concatenate()([x, x_dilated]) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)     
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)     
    x = Concatenate()([x, x_c])    
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model



def create_multifilter_segmentation_model10(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv3D
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    #laplacian = Laplacian()(inputs) 

    slope = Conv2D(16, (3,3), activation="relu", padding="same")(slope) 
    reflectance = Conv2D(16, (3,3), activation="relu", padding="same")(reflectance)
    usc = Conv2D(16, (3,3), activation="relu", padding="same")(usc)
    shape_index = Conv2D(16, (3,3), activation="relu", padding="same")(shape_index)



    x_c = Concatenate()([slope, reflectance, usc, shape_index]) #, laplacian
    #x_c = tf.reshape(x_c, (None, width, height, 1))
    x = Conv3D(16, 3, activation="relu", padding="same")(x_c)
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    #x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x)    
    #x = Concatenate()([x, x_dilated]) 
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)  
    #x = Conv2D(32, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(16, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(16, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(32, (1,1), activation="relu", padding="same")(x) 
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)  
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(x)  
    #x = Conv2D(128, (3,3), activation="relu", padding="same")(x)  
    #x = Conv2D(128, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(128, (3,3), activation="relu", padding="same")(x) 
    #x = Conv2D(128, (3,3), activation="relu", padding="same")(x)  
    x = Concatenate()([x, x_c])    
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model




def create_multifilter_segmentation_model11(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  

    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 

    #slope = Conv2D(16, (3,3), activation="relu", padding="same")(slope) 
    #reflectance = Conv2D(16, (3,3), activation="relu", padding="same")(reflectance)
    #usc = Conv2D(16, (3,3), activation="relu", padding="same")(usc)
    #shape_index = Conv2D(16, (3,3), activation="relu", padding="same")(shape_index)
    x_c = Concatenate()([slope, reflectance, usc, shape_index, laplacian]) #
    x_c = Conv2D(10, (3,3), activation="relu", padding="same")(x_c)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = AveragePooling2D((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = AveragePooling2D((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model


def create_multifilter_segmentation_model12(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  

    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 

    
    x_c = Concatenate()([slope, reflectance, usc, shape_index, laplacian]) #
    x_c = Conv2D(10, (3,3), activation="relu", padding="same")(x_c)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = AveragePooling2D((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = AveragePooling2D((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model



def create_multifilter_segmentation_model12_beta(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature
    optimizer = "adam"
    loss = f_beta_loss

    inputs = Input(shape=(width, height, 1))  

    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 

    
    x_c = Concatenate()([slope, reflectance, usc, shape_index, laplacian]) #
    x_c = Conv2D(10, (3,3), activation="relu", padding="same")(x_c)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = AveragePooling2D((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = AveragePooling2D((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=[f_beta, f1, "accuracy"])  
    #model.summary()
    return model



def create_evaluation_model12(width, height): 
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Add
    from custom_layers import Rotate, Divide
    seg_model = create_multifilter_segmentation_model12(width, height)
    seg_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-18:13-35-35epoch_00060_copy.h5")    
    seg_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))   
    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 
    
    x = seg_model(inputs)
    x1 = seg_model(x_rotated1) 
    x2 = seg_model(x_rotated2) 
    x3 = seg_model(x_rotated3)
    x1 = Rotate(k=3)(x1)       
    x2 = Rotate(k=2)(x2)   
    x3 = Rotate(k=1)(x3) 

    x = Add()([x, x1, x2, x3]) 
    outputs = Divide(divisor=4.)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])     
    return model






def create_multifilter_segmentation_model13(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature, MeanCurvature, SumOfGradients, MinimalCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  

    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 
    sum_of_gradients = SumOfGradients()(inputs) 
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)

    
    x_c = Concatenate()([slope, reflectance, usc, shape_index, laplacian, sum_of_gradients, minimal_curvature, mean_curvature]) #
    x_c = Conv2D(10, (3,3), activation="relu", padding="same")(x_c)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = AveragePooling2D((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = AveragePooling2D((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[f1, "accuracy"]) # f1_loss
    #model.summary()
    return model



def create_multifilter_segmentation_model14(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    #from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature, MeanCurvature, SumOfGradients, MinimalCurvature
    from custom_layers import SlopeTrainable, LaplacianTrainable, ShapeIndexTrainable, ReflectanceTrainable, UnsphericityCurvatureTrainable, MeanCurvatureTrainable, SumOfGradientsTrainable, MinimalCurvatureTrainable
    optimizer = "adam"
    #loss = f1_loss

    pooling = MaxPooling2D


    inputs = Input(shape=(width, height, 1))  
    """
    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    #shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 
    sum_of_gradients = SumOfGradients()(inputs) 
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)
    """

    slope = SlopeTrainable()(inputs)   
    reflectance = ReflectanceTrainable()(inputs)      
    usc = UnsphericityCurvatureTrainable()(inputs)
    #shape_index = ShapeIndexTrainable()(inputs)    
    laplacian = LaplacianTrainable()(inputs) 
    sum_of_gradients = SumOfGradientsTrainable()(inputs) 
    minimal_curvature = MinimalCurvatureTrainable()(inputs) 
    mean_curvature = MeanCurvatureTrainable()(inputs)
    
    #x_c = Concatenate()([reflectance, sum_of_gradients, minimal_curvature, mean_curvature]) #slope, usc, shape_index, laplacian, 
    x_c = Concatenate()([slope, reflectance, usc, laplacian, sum_of_gradients, minimal_curvature, mean_curvature]) # , , shape_index


    a = Conv2D(64, (1,1), activation="relu", padding="same")(x_c)
    a = Conv2D(64, (3,3), activation="relu", padding="same")(a)

    b = Conv2D(64, (1,1), activation="relu", padding="same")(x_c)
    b = Conv2D(64, (5,5), activation="relu", padding="same")(b)

    c = Conv2D(128, (1,1), activation="relu", padding="same")(x_c)

    x1 = Concatenate()([a, b, c])
    x1 = Conv2D(32, (1,1), activation="relu", padding="same")(x1)

    #x_c = Conv2D(128, (3,3), activation="relu", padding="same")(x_c)
    #x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    #x1 = Conv2D(32, (1,1), activation="relu", padding="same")(x_c)
    #x1 = BatchNormalization()(x1)    
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x2 = BatchNormalization()(x2) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"]) # f_beta_loss
    #model.summary()
    return model




def create_multifilter_segmentation_model16(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature, MeanCurvature, SumOfGradients, MinimalCurvature
    #from custom_layers import SlopeTrainable, LaplacianTrainable, ShapeIndexTrainable, ReflectanceTrainable, UnsphericityCurvatureTrainable, MeanCurvatureTrainable, SumOfGradientsTrainable, MinimalCurvatureTrainable
    optimizer = "adam"
    #loss = f1_loss

    pooling = MaxPooling2D


    inputs = Input(shape=(width, height, 1))  
    
    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 
    sum_of_gradients = SumOfGradients()(inputs) 
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)
    
    """
    slope = SlopeTrainable()(inputs)   
    reflectance = ReflectanceTrainable()(inputs)      
    usc = UnsphericityCurvatureTrainable()(inputs)
    #shape_index = ShapeIndexTrainable()(inputs)    
    laplacian = LaplacianTrainable()(inputs) 
    sum_of_gradients = SumOfGradientsTrainable()(inputs) 
    minimal_curvature = MinimalCurvatureTrainable()(inputs) 
    mean_curvature = MeanCurvatureTrainable()(inputs)
    """
    #x_c = Concatenate()([reflectance, sum_of_gradients, minimal_curvature, mean_curvature]) #slope, usc, shape_index, laplacian, 
    x_c = Concatenate()([slope, reflectance, usc, shape_index, laplacian, sum_of_gradients, minimal_curvature, mean_curvature]) # , 


    a = Conv2D(64, (1,1), activation="relu", padding="same")(x_c)
    a = Conv2D(64, (3,3), activation="relu", padding="same")(a)

    b = Conv2D(64, (1,1), activation="relu", padding="same")(x_c)
    b = Conv2D(64, (5,5), activation="relu", padding="same")(b)

    c = Conv2D(128, (1,1), activation="relu", padding="same")(x_c)

    x1 = Concatenate()([a, b, c])
    x1 = Conv2D(32, (1,1), activation="relu", padding="same")(x1)
    x1 = BatchNormalization()(x1)
    
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    #x2 = BatchNormalization()(x2) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x) 
    #x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"]) # f_beta_loss
    #model.summary()
    return model


def create_multifilter_segmentation_model15(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Laplacian, ShapeIndex, Reflectance, Rotate, UnsphericityCurvature, MeanCurvature, SumOfGradients, MinimalCurvature
    optimizer = "adam"
    loss = f1_loss
    pooling = MaxPooling2D
    inputs = Input(shape=(width, height, 1))  
    slope = Slope()(inputs)   
    reflectance = Reflectance()(inputs)      
    usc = UnsphericityCurvature()(inputs)
    #shape_index = ShapeIndex()(inputs)    
    laplacian = Laplacian()(inputs) 
    sum_of_gradients = SumOfGradients()(inputs) 
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)

    
    #x_c = Concatenate()([reflectance, sum_of_gradients, minimal_curvature, mean_curvature]) #slope, usc, shape_index, laplacian, 
    x_concat = Concatenate()([slope, reflectance, usc, laplacian, sum_of_gradients, minimal_curvature, mean_curvature])

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x_concat)    
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x_concat)    
    x1 = Concatenate()([x, x_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    
    x = pooling((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = pooling((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    
    x = pooling((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    
    x = pooling((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)  

    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)   

    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    model.summary()
    return model





def create_multifilter_segmentation_model17(width, height): 
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate   
    from tensorflow.keras.models import Model
    from custom_layers import MeanCurvature, MinimalCurvature
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  
    
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)


    inputs1 = minimal_curvature 
    inputs2 = mean_curvature

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs1)
    y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, x_dilated, y, y_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    #x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    #x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    #x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    return model



def filter_block(inputs1, inputs2): 
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs1)
    y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    filtered = Concatenate()([x, x_dilated, y, y_dilated])
    return  Conv2D(64, (1,1), activation="relu", padding="same")(filtered)



def create_multifilter_segmentation_model18(width, height): 
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate   
    from tensorflow.keras.models import Model
    from custom_layers import MeanCurvature, MinimalCurvature, SumOfGradients, Laplacian
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  
    
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)

    pseudo_slope = SumOfGradients()(inputs) 
    laplacian = Laplacian()(inputs) 

    filter1 = filter_block(minimal_curvature, mean_curvature) 
    filter2 = filter_block(pseudo_slope, laplacian)  

    x1 = Concatenate()([filter1, filter2])
    
    

    #x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    #x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    #x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    #x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    return model




def create_multifilter_segmentation_model19(width, height): 
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate   
    from tensorflow.keras.models import Model
    from custom_layers import SumOfGradients, Laplacian
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  
    
    inputs1 = SumOfGradients()(inputs) 
    inputs2 = Laplacian()(inputs) 


    #inputs1 = minimal_curvature 
    #inputs2 = mean_curvature

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs1)
    y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, x_dilated, y, y_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    #x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    #x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    #x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    return model



def create_multifilter_segmentation_model20(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    optimizer = "adam"
    loss = f1_loss

    inputs = Input(shape=(width, height, 1))  
    
    laplacian = Laplacian2()(inputs) 
    nabla = SumOfGradients()(inputs)
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)

    filters = Concatenate()([laplacian, nabla, minimal_curvature, mean_curvature])     
    
    x_c = Conv2D(10, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = AveragePooling2D((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = AveragePooling2D((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model



def filter_block2(inputs): 
    from tensorflow.keras.layers import Concatenate
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    laplacian = Laplacian2()(inputs) 
    nabla = SumOfGradients()(inputs)
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)
    return Concatenate()([laplacian, nabla, minimal_curvature, mean_curvature])


def create_multifilter_segmentation_model21(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    optimizer = "adam"
    loss = f1_loss
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy", precision, recall])  
    #model.summary()
    return model


def create_multifilter_segmentation_model21_f1_2(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    optimizer = "adam"  
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_2_loss, metrics=[tversky_index, "accuracy", precision, recall])  
    #model.summary()
    return model


def create_multifilter_segmentation_model21_f_beta(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    optimizer = "adam"  
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f_beta_loss, metrics=[f1, "accuracy", precision, recall])  
    #model.summary()
    return model

def create_multifilter_segmentation_model21_tversky(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    
    optimizer = "adam"    
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=tversky_loss, metrics=[f1, "accuracy"])  
    
    return model


def create_multifilter_segmentation_model21_weighed_tversky(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    
    optimizer = "adam"    
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=mcc_loss, metrics=[f1, "accuracy"])  #weighed_tversky_loss
    
    return model



def create_multifilter_segmentation_model21_fowlkes_mallows(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    
    optimizer = "adam"    
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=fowlkes_mallows_loss, metrics=[f1, "accuracy"])  #weighed_tversky_loss
    
    return model




def create_multifilter_segmentation_model22(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    optimizer = tf.keras.optimizers.Adam()
    loss = f1_loss
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block2(inputs)
    rotated1 = filter_block2(rotated1) 
    rotated2 = filter_block2(rotated2) 
    rotated3 = filter_block2(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model






def filter_block3(inputs): 
    from tensorflow.keras.layers import Concatenate
    from custom_layers import Rotate, Slope, ShapeIndex, Reflectance, UnsphericityCurvature, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    laplacian = Laplacian2()(inputs) 
    nabla = SumOfGradients()(inputs)
    minimal_curvature = MinimalCurvature()(inputs) 
    mean_curvature = MeanCurvature()(inputs)

    slope = Slope()(inputs) 
    shape_index = ShapeIndex()(inputs) 
    reflectance = Reflectance()(inputs) 
    usc = UnsphericityCurvature()(inputs)

    return Concatenate()([laplacian, nabla, minimal_curvature, mean_curvature, slope, shape_index, reflectance, usc])


def create_multifilter_segmentation_model23(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, AveragePooling2D    
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian2, MeanCurvature, MinimalCurvature, SumOfGradients
    optimizer = "adam"
    loss = f1_loss
    pooling = MaxPooling2D

    inputs = Input(shape=(width, height, 1))      
    rotated1 = Rotate(k=1)(inputs) 
    rotated2 = Rotate(k=2)(inputs) 
    rotated3 = Rotate(k=3)(inputs) 

    rotated0 = filter_block3(inputs)
    rotated1 = filter_block3(rotated1) 
    rotated2 = filter_block3(rotated2) 
    rotated3 = filter_block3(rotated3) 

    rotated1 = Rotate(k=3)(rotated1) 
    rotated2 = Rotate(k=2)(rotated2) 
    rotated3 = Rotate(k=1)(rotated3) 
    
    filters = Concatenate()([rotated0, rotated1, rotated2, rotated3]) 

    x_c = Conv2D(16, (3,3), activation="relu", padding="same")(filters)
    x1 = Conv2D(64, (3,3), activation="relu", padding="same")(x_c) 
    x = pooling((2,2), padding="same")(x1)
    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = pooling((2,2), padding="same")(x2)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x2])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = Conv2D(64, (1,1), activation="relu", padding="same")(x)   
    x = UpSampling2D((2,2))(x)  
    x = Concatenate()([x, x1])  
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Dropout(0.2)(x)
    #x = Concatenate()([x, x_c])    
    #x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (1,1), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (1,1), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    #model.summary()
    return model

























def create_multifilter_detection_model(width=64, height=64): 
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  

    pretrained_model = create_multifilter_segmentation_model4(width, height)
    #pretrained_model.load_weights("checkpoints/reflectance_multiscale_transferable_model_reflectance_2021-3-6:9-33-47epoch_00080.h5" )     
    pretrained_model.load_weights("checkpoints/multifilter_detection/multifilter_segmentation_model_2021-5-6:0-40-21epoch_00100.h5")
    pretrained_model.trainable = True
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)
    y = Laplacian()(inputs)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    x = Concatenate()([x, y])
    # reduce size by pooling
    p1 = MaxPooling2D((2,2), padding="same")(x)
    p2 = MaxPooling2D((2,2), padding="same")(p1)
    p3 = AveragePooling2D((2,2), padding="same")(p2)
    #p4 = AveragePooling2D((2,2), padding="same")(p3)

    c = Conv2D(1, (3,3), activation="relu", padding="same")(p3)
    c = UpSampling2D((8,8))(c)

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)      
    classifier = Conv2D(1, (3,3), activation="sigmoid", padding="same")(c)

    # pixelwise regression of radius
    regressor = Conv2D(1, (3,3), activation="linear", padding="same")(c)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=["binary_crossentropy", "mse"])
    return model


def create_multifilter_detection_model2(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  

    pretrained_model = create_multifilter_segmentation_model5(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_detection/multifilter_segmentation_model_2021-5-7:14-21-19epoch_00100.h5")
    pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    y = Laplacian()(inputs)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    x = Concatenate()([x, y])
    # reduce size by pooling
    p1 = MaxPooling2D((2,2), padding="same")(x)
    p2 = MaxPooling2D((2,2), padding="same")(p1)
    p3 = AveragePooling2D((2,2), padding="same")(p2)
    #p4 = AveragePooling2D((2,2), padding="same")(p3)

    c = Conv2D(1, (3,3), activation="relu", padding="same")(p3)
    c = UpSampling2D((8,8))(c)

    c = Concatenate()([c, y])

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(32, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(32, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, f1_loss])  #"mse"
    return model


def create_multifilter_detection_model3(width=64, height=64): 
    
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  
    from custom_metrics import threshold

    pretrained_model = create_multifilter_segmentation_model5(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_detection/multifilter_segmentation_model_2021-5-7:14-21-19epoch_00100.h5")
    pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    #y = Laplacian()(inputs)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    #x = Concatenate()([x, y])
    # reduce size by pooling
    #p1 = MaxPooling2D((2,2), padding="same")(x)
    #p2 = MaxPooling2D((2,2), padding="same")(p1)
    #p3 = AveragePooling2D((2,2), padding="same")(p2)
    #p4 = AveragePooling2D((2,2), padding="same")(p3)

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    #c = UpSampling2D((8,8))(c)

    #c = Concatenate()([c, y])

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, threshold])  #f1_loss "mse"     
    return model



def create_multifilter_detection_model4(width=64, height=64): 
    
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  
    from custom_metrics import threshold

    pretrained_model = create_multifilter_segmentation_model6(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_detection/usc/multifilter_segmentation_model_2021-5-8:18-19-37epoch_00100.h5")
    pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    #y = Laplacian()(inputs)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    #x = Concatenate()([x, y])
    # reduce size by pooling
    #p1 = MaxPooling2D((2,2), padding="same")(x)
    #p2 = MaxPooling2D((2,2), padding="same")(p1)
    #p3 = AveragePooling2D((2,2), padding="same")(p2)
    #p4 = AveragePooling2D((2,2), padding="same")(p3)

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    #c = UpSampling2D((8,8))(c)

    #c = Concatenate()([c, y])

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, threshold])  #f1_loss "mse"     
    return model




def create_multifilter_detection_model5(width=64, height=64): 
    
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  
    from custom_metrics import positive_MAE

    pretrained_model = create_multifilter_segmentation_model6(width, height)         
    #pretrained_model.load_weights("checkpoints/multifilter_detection/usc/multifilter_segmentation_model_2021-5-8:18-19-37epoch_00100.h5")
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-8:18-19-37epoch_00100_copy.h5")
    pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    #y = Laplacian()(inputs)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    #x = Concatenate()([x, y])
    # reduce size by pooling
    #p1 = MaxPooling2D((2,2), padding="same")(x)
    #p2 = MaxPooling2D((2,2), padding="same")(p1)
    #p3 = AveragePooling2D((2,2), padding="same")(p2)
    #p4 = AveragePooling2D((2,2), padding="same")(p3)

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    #c = UpSampling2D((8,8))(c)

    #c = Concatenate()([c, y])

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MAE])  #f1_loss "mse"     
    return model




def create_multifilter_detection_model6(width=64, height=64): 
    # no shape_index
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  
    from custom_metrics import positive_MSE

    pretrained_model = create_multifilter_segmentation_model7(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_detection/no_shape_index/multifilter_segmentation_model_2021-5-11:18-0-41epoch_00100_copy.h5")
    pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    #y = Laplacian()(inputs)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    #x = Concatenate()([x, y])
    # reduce size by pooling
    #p1 = MaxPooling2D((2,2), padding="same")(x)
    #p2 = MaxPooling2D((2,2), padding="same")(p1)
    #p3 = AveragePooling2D((2,2), padding="same")(p2)
    #p4 = AveragePooling2D((2,2), padding="same")(p3)

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    #c = UpSampling2D((8,8))(c)

    #c = Concatenate()([c, y])

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])  #f1_loss "mse"     
    return model




def create_multifilter_detection_model7(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  
    from custom_metrics import positive_MSE

    pretrained_model = create_multifilter_segmentation_model7(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-13:0-50-33epoch_00100_copy.h5")
    pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    #y = Laplacian()(inputs)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model8(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian  
    from custom_metrics import positive_MSE

    pretrained_model = create_multifilter_segmentation_model7(width, height)         
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-13:0-50-33epoch_00100_copy _for_transfer_learning.h5")
    pretrained_model.layers.pop()
    #pretrained_model.trainable = True
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)

    #y = Laplacian()(inputs)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)   
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model9(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    #from custom_layers import Laplacian  
    from custom_metrics import positive_MSE

    pretrained_model = create_multifilter_segmentation_model8(width, height)         
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-13:0-50-33epoch_00100_copy _for_transfer_learning.h5")
    pretrained_model.layers.pop()
    #pretrained_model.trainable = True
    optimizer = "adam"

    inputs = Input(shape=(None, None, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)
    
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model


def create_multifilter_detection_model10(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    #from custom_layers import Laplacian  
    from custom_metrics import positive_MSE

    pretrained_model = create_multifilter_segmentation_model8(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:1-48-1epoch_00040_copy.h5")
    #pretrained_model = pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)
    #x = m.layers[-1].output 
    
    y = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    #classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model11(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE

    #pretrained_model = create_multifilter_segmentation_model8(width, height)         
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:1-48-1epoch_00100.h5")
    pretrained_model = create_multifilter_segmentation_model9(64, 64)    
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:19-28-45epoch_00100.h5")
    #pretrained_model = pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    #laplacian = Laplacian()(inputs) 


    x_c = Concatenate()([slope, reflectance, usc, shape_index]) #, laplacian
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)
    x1 = pretrained_model(x_rotated1) 
    x2 = pretrained_model(x_rotated2) 
    x3 = pretrained_model(x_rotated3)
    x1 = Rotate(k=3)(x1)       
    x2 = Rotate(k=2)(x2)   
    x3 = Rotate(k=1)(x3) 

    x = Add()([x, x1, x2, x3]) 
    x = Divide(divisor=4.)(x)

    x = Concatenate()([x_c, x])
    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model





def create_multifilter_detection_model12(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE

    #pretrained_model = create_multifilter_segmentation_model8(width, height)         
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:1-48-1epoch_00100.h5")
    #pretrained_model = create_multifilter_segmentation_model9(64, 64)    
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:19-28-45epoch_00100.h5")
    pretrained_model = create_multifilter_segmentation_model7(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-13:0-50-33epoch_00100_copy2.h5")
    #pretrained_model = pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    reflectance1 = Reflectance()(inputs)
    x_reflected1 = Reflectance()(x_rotated1)       
    x_reflected2 = Reflectance()(x_rotated2)   
    x_reflected3 = Reflectance()(x_rotated3) 

    reflectance2 = Rotate(k=3)(x_reflected1)       
    reflectance3 = Rotate(k=2)(x_reflected2)   
    reflectance4 = Rotate(k=1)(x_reflected3) 
    reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    usc = UnsphericityCurvature()(inputs)
    shape_index = ShapeIndex()(inputs)    
    #laplacian = Laplacian()(inputs) 


    x_c = Concatenate()([slope, reflectance, usc, shape_index]) #, laplacian
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)
    x1 = pretrained_model(x_rotated1) 
    x2 = pretrained_model(x_rotated2) 
    x3 = pretrained_model(x_rotated3)
    x1 = Rotate(k=3)(x1)       
    x2 = Rotate(k=2)(x2)   
    x3 = Rotate(k=1)(x3) 

    x = Add()([x, x1, x2, x3]) 
    x = Divide(divisor=4.)(x)

    x = Concatenate()([x_c, x])
    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model13(width=64, height=64): 
    # no shape_index, bigger training segmentation data
    # with usc
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE

    #pretrained_model = create_multifilter_segmentation_model8(width, height)         
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:1-48-1epoch_00100.h5")
    #pretrained_model = create_multifilter_segmentation_model9(64, 64)    
    #pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:19-28-45epoch_00100.h5")
    pretrained_model = create_multifilter_segmentation_model11(width, height)         
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-17:1-10-20epoch_00120_copy.h5")
    #pretrained_model = pretrained_model.layers.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    #slope = Slope()(inputs) 

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 

    #reflectance1 = Reflectance()(inputs)
    #x_reflected1 = Reflectance()(x_rotated1)       
    #x_reflected2 = Reflectance()(x_rotated2)   
    #x_reflected3 = Reflectance()(x_rotated3) 

    #reflectance2 = Rotate(k=3)(x_reflected1)       
    #reflectance3 = Rotate(k=2)(x_reflected2)   
    #reflectance4 = Rotate(k=1)(x_reflected3) 
    #reflectance = Concatenate()([reflectance1, reflectance2, reflectance3, reflectance4])
    
    #usc = UnsphericityCurvature()(inputs)
    #shape_index = ShapeIndex()(inputs)    
    #laplacian = Laplacian()(inputs) 


    #x_c = Concatenate()([slope, reflectance, usc, shape_index]) #, laplacian
    # get features from pretrained segmentation model and the original image
    x = pretrained_model(inputs)
    x1 = pretrained_model(x_rotated1) 
    x2 = pretrained_model(x_rotated2) 
    x3 = pretrained_model(x_rotated3)
    x1 = Rotate(k=3)(x1)       
    x2 = Rotate(k=2)(x2)   
    x3 = Rotate(k=1)(x3) 

    x = Add()([x, x1, x2, x3]) 
    x = Divide(divisor=4.)(x)

    #x = Concatenate()([x_c, x])
    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model





def create_multifilter_detection_model14(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model12(64, 64)         
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-18:13-35-35epoch_00060_copy.h5")    
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     

    x_rotated1 = Rotate(k=1)(inputs)       
    x_rotated2 = Rotate(k=2)(inputs)   
    x_rotated3 = Rotate(k=3)(inputs) 
    
    x = pretrained_model(inputs)
    x1 = pretrained_model(x_rotated1) 
    x2 = pretrained_model(x_rotated2) 
    x3 = pretrained_model(x_rotated3)
    x1 = Rotate(k=3)(x1)       
    x2 = Rotate(k=2)(x2)   
    x3 = Rotate(k=1)(x3) 

    x = Add()([x, x1, x2, x3]) 
    x = Divide(divisor=4.)(x)

    #x = Concatenate()([x_c, x])
    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model15(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model19(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:18-24-11epoch_00100_copy.h5")

    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    x = pretrained_model(inputs)  

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model16(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-28:14-45-48epoch_00080_copy.h5")
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    #inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model.layers[-2].output

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=pretrained_model.inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model





def create_multifilter_detection_model17(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-28:14-45-48epoch_00080_copy2.h5")
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model19(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-6-3:18-42-22epoch_00100.h5")
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model18(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-6-1:2-37-59epoch_00100_copy.h5")
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model






def create_multifilter_detection_model20(width=64, height=64):     
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_2021-6-2:11-50-12epoch_00060_copy.h5")
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model




def create_multifilter_detection_model21(width=64, height=64): 
    # D1    
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-6-7:19-6-28epoch_00100_copy.h5")
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model





def create_multifilter_detection_model22(width=64, height=64):  
    # D2   
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21(64, 64)             
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-6-8:22-15-5epoch_00080_copy.h5")    
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[f1_loss, positive_MSE])     
    return model






def create_multifilter_detection_model23_weighed(width=64, height=64):  
    # D2   
    from tensorflow.keras.layers import Input, Conv2D, Dropout, Flatten, Concatenate, Add
    from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, MaxPooling2D
    from tensorflow.keras.models import Model
    from custom_layers import Slope, Reflectance, UnsphericityCurvature, ShapeIndex, Rotate, Divide 
    from custom_metrics import positive_MSE
    
    pretrained_model = create_multifilter_segmentation_model21_f_beta(64, 64)       
    pretrained_model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-7-3:15-4-39epoch_00100_copy.h5")    
                                 
    #pretrained_model = pretrained_model.pop()
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    #x = pretrained_model(inputs)  
    x = pretrained_model(inputs)

    y = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y)    

    c = Conv2D(32, (1,1), activation="relu", padding="same")(y)    

    # pixelwise classification {0, 1} (middle of gravemound = 1, else = 0)   
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(c)  
    classifier = Conv2D(64, (3,3), activation="linear", padding="same")(classifier)  # remove for comparison
    classifier = Conv2D(1, (1,1), activation="sigmoid", padding="same", name="classifier_out")(classifier)

    # pixelwise regression of radius
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(c)
    regressor = Conv2D(64, (3,3), activation="linear", padding="same")(regressor)    # remove for comparison
    regressor = Conv2D(1, (1,1), activation="linear", padding="same", name="regressor_out")(regressor)

    model = Model(inputs=inputs, outputs=[classifier, regressor])
    model.compile(optimizer=optimizer, loss=[tversky_loss, positive_MSE])     
    return model
























def create_multiscale_segmentation_model4(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    #he = "glorot_uniform"
    inputs = Input(shape=(width, height, 1))     
    #x = Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=he)(inputs) 
    x_dilated1 = Conv2D(32, (3,3), dilation_rate=1, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_concat = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model




def create_multiscale_segmentation_model2(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    inputs = Input(shape=(width, height, 1)) 
    he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    #x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs) 
    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(inputs)    
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_concat = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    
    #x_concat = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_concat)
    #x = Dropout(0.2)(x)
    x = MaxPooling2D((2,2), padding="same")(x) 
    #x = BatchNormalization()(x)


    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 
    #x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)   
    #x = BatchNormalization()(x)  
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 

    x = UpSampling2D((2,2))(x)   
    #x = BatchNormalization()(x)  
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss)

    return model



def rotation(x): 
    # rotation counterclockwise of 2d-tensor.    
    from keras.backend import transpose, reverse 
    x = transpose(x)
    x = reverse(x, axes=1)  #
    x = reverse(x, axes=1)  #
    return x


def create_rotation_reflectance_model(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate, Average
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Reflectance
    from tensorflow.contrib.image import rotate
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    he = "glorot_uniform"
    inputs = Input(shape=(width, height, 1)) 
    #x = tf.reshape(inputs, (width, height))    
    x = inputs
    x_rotated1 = Rotate(k=1)(x)       
    x_rotated2 = Rotate(k=2)(x)   
    x_rotated3 = Rotate(k=3)(x) 

    x = Reflectance()(x)
    x_rotated1 = Reflectance()(x)       
    x_rotated2 = Reflectance()(x)   
    x_rotated3 = Reflectance()(x) 

    x_rotated1 = Rotate(k=3)(x)       
    x_rotated2 = Rotate(k=2)(x)   
    x_rotated3 = Rotate(k=1)(x) 

    x = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    x_rotated1 = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_rotated1)
    x_rotated2 = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_rotated2)
    x_rotated3 = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_rotated3)

    #x_concat = Concatenate()([x, x_rotated1, x_rotated2, x_rotated3])   
    x_concat = Average()([x, x_rotated1, x_rotated2, x_rotated3])   
    
    x_dilated2 = Conv2D(16, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_dilated3 = Conv2D(16, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_dilated4 = Conv2D(16, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(x_concat)
    x_concat2 = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    #x_concat3 = Concatenate()([x_concat, x_concat2])

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_concat2)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    #x = Concatenate()([x_concat2, x])
    x = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[precision, recall, f1, "accuracy"]) #"poisson"focal_tversky_loss"binary_crossentropy" f1_loss f1_loss_by_confusion_matrix

    return model


def create_rotation_laplacian_model(batch_size, width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, BatchNormalization, Concatenate
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Laplacian
    from tensorflow.contrib.image import rotate
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #he = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    he = "glorot_uniform"
    inputs = Input(shape=(width, height, 1)) 
    #x = tf.reshape(inputs, (width, height))    
    x = inputs
    x_rotated1 = Rotation(k=1)(x)       
    x_rotated2 = Rotation(k=2)(x)   
    x_rotated3 = Rotation(k=3)(x) 

    x_rotated1 = Laplacian()(x)       
    x_rotated2 = Laplacian()(x)   
    x_rotated3 = Laplacian()(x) 

    x_rotated1 = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_rotated1)
    x_rotated2 = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_rotated2)
    x_rotated3 = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_rotated3)


    #x = tf.reshape(x, (-1, width, height, 1))
    x = Conv2D(16, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)
    x_concat = Concatenate()([x, x_rotated1, x_rotated2, x_rotated3]) 
    
    x_dilated2 = Conv2D(16, (3,3), dilation_rate=2, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated3 = Conv2D(16, (3,3), dilation_rate=3, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_dilated4 = Conv2D(16, (3,3), dilation_rate=4, activation="relu", kernel_initializer=he, padding="same")(inputs)
    x_concat2 = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    x_concat3 = Concatenate()([x_concat, x_concat2])

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x_concat3)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", kernel_initializer=he, padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", kernel_initializer=he, padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss)

    return model






def create_transfer_segmentation_model(width=64, height=64): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.layers import UpSampling2D, BatchNormalization, Concatenate, Average, Reshape
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Divide, Threshold   

    pretrained_model = create_multiscale_model(width, height)
    #pretrained_model.load_weights("checkpoints/reflectance_multiscale_transferable_model_reflectance_2021-3-6:9-33-47epoch_00080.h5" )     
    pretrained_model.load_weights("checkpoints/multiscale_transferable_model_reflectance_2021-3-10:22-36-15epoch_00100.h5")
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    x = inputs
    x_rotated1 = Rotate(k=1)(x)       
    x_rotated2 = Rotate(k=2)(x)   
    x_rotated3 = Rotate(k=3)(x)     

    x_rotated0 = pretrained_model(x)
    x_rotated1 = pretrained_model(x_rotated1)       
    x_rotated2 = pretrained_model(x_rotated2)     
    x_rotated3 = pretrained_model(x_rotated3)  

    x_rotated1 = Rotate(k=3)(x_rotated1)       
    x_rotated2 = Rotate(k=2)(x_rotated2)   
    x_rotated3 = Rotate(k=1)(x_rotated3)

    y = Add()([x_rotated0, x_rotated1, x_rotated2, x_rotated3]) 
    outputs = Divide(divisor=4.)(y)
    
    
    """
    y = Conv2D(16, (3,3), activation="relu", padding="same")(y)
    y = MaxPooling2D((2,2), padding="same")(y)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y) 
    y = UpSampling2D()(y)
    y = Conv2D(16, (3,3), activation="relu", padding="same")(y)
    """
    #outputs = Threshold()(y)
    #outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(y)
    #y = Flatten()(y) 
    #y = Dense(2, activation="relu")(y) 
    #outputs = Dense(width*height, activation="sigmoid")(y) 
    #outputs = Reshape((-1, width, height, 1))(outputs) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[precision, recall, "accuracy"])
    return model






def create_multiscale_model(width=64, height=64): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.layers import UpSampling2D, BatchNormalization, Concatenate, Average
    from tensorflow.keras.models import Model
    from custom_layers import Reflectance       
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    x = Reflectance()(inputs)

    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", padding="same")(x)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", padding="same")(x)
    x_concat = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x_concat)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])

    return model




def create_multiscale_laplace_model(width=64, height=64): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.layers import UpSampling2D, BatchNormalization, Concatenate, Average
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian      
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1))     
    
    x = Laplacian()(inputs)

    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", padding="same")(x)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", padding="same")(x)
    x_concat = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x_concat)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[precision, recall, "accuracy"])

    return model




def create_multiscale_laplace_model_experimental(width=64, height=64): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.layers import UpSampling2D, BatchNormalization, Concatenate, Average
    from tensorflow.keras.models import Model
    from custom_layers import Laplacian      
    optimizer = "adam"    
    inputs = Input(shape=(width, height, 1))     
    
    x = Laplacian()(inputs)

    x_dilated2 = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(x)
    x_dilated3 = Conv2D(32, (3,3), dilation_rate=3, activation="relu", padding="same")(x)
    x_dilated4 = Conv2D(32, (3,3), dilation_rate=4, activation="relu", padding="same")(x)
    x_concat = Concatenate()([x_dilated2, x_dilated3, x_dilated4])
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x_concat)
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x) 

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)    
    
    x = UpSampling2D((2,2))(x)     
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Concatenate()([x_concat, x])
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=["accuracy"])  #sklearn_mcc_loss, sklearn_auc_loss, , mcc

    return model








def create_transfer_segmentation_model_laplace(width=64, height=64): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.layers import UpSampling2D, BatchNormalization, Concatenate, Average, Reshape
    from tensorflow.keras.layers import Activation
    from tensorflow.keras import activations
    from tensorflow.keras.models import Model
    from custom_layers import Rotate, Divide, Threshold   

    #pretrained_model = create_multiscale_laplace_model(width, height)
    #pretrained_model.load_weights("checkpoints/multiscale_transferable_model_laplacian_2021-3-7:19-16-39epoch_00100.h5") 
    #pretrained_model.load_weights("checkpoints/multiscale_transferable_model_smaller_test_set_laplacian_2021-3-8:14-8-28epoch_00100.h5")
    #pretrained_model = create_multiscale_laplace_model_experimental(width, height)
    #pretrained_model.load_weights("checkpoints/multiscale_transferable_model_smaller_test_set_laplacian_2021-3-9:1-57-36epoch_00060.h5")
    pretrained_model = create_multiscale_laplace_model_experimental(width, height)
    pretrained_model.load_weights("checkpoints/multiscale_transferable_model_smaller_test_set_laplacian_2021-3-10:18-18-29epoch_00070.h5")
    pretrained_model.trainable = False
    optimizer = "adam"

    inputs = Input(shape=(width, height, 1)) 
    x = inputs
    x_rotated1 = Rotate(k=1)(x)       
    x_rotated2 = Rotate(k=2)(x)   
    x_rotated3 = Rotate(k=3)(x)     

    x_rotated0 = pretrained_model(x)
    x_rotated1 = pretrained_model(x_rotated1)       
    x_rotated2 = pretrained_model(x_rotated2)     
    x_rotated3 = pretrained_model(x_rotated3)  

    x_rotated1 = Rotate(k=3)(x_rotated1)       
    x_rotated2 = Rotate(k=2)(x_rotated2)   
    x_rotated3 = Rotate(k=1)(x_rotated3)

    y = Add()([x_rotated0, x_rotated1, x_rotated2, x_rotated3]) 
    outputs = Divide(divisor=4.)(y)
        
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[precision, recall, "accuracy"])
    return model




def create_single_input_model(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    optimizer = "adam"
    loss = f1_loss
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs)
    x1 = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=inputs, outputs=outputs)    
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1]) 
    return model











def summary_model8(): 
    model = create_conv_segmentation_model8(64, 64)
    model.summary()


def create_multi_input_model(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add
    from tensorflow.keras.models import Model
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    #optimizer = "adam"
    inputs1 = Input(shape=(width, height, 1)) 
    inputs2 = Input(shape=(width, height, 1))

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1) 
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2,2), padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x)     
    x = Conv2D(32, (3,3), activation= "relu", padding="same")(x)     

    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2) 
    y = Dropout(0.2)(y)
    y = MaxPooling2D((2,2), padding="same")(y) 
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y) 
    y = MaxPooling2D((2,2), padding="same")(y)     
    y = Conv2D(32, (3,3), activation= "relu", padding="same")(y)     

    z = Add()([x, y])     

    z = UpSampling2D((2,2))(z) 
    z = Conv2D(32, (3,3), activation="relu", padding="same")(z) 
    z = UpSampling2D((2,2))(z) 

    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(z) 

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    return model





def create_multi_input_model2(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #optimizer="sgd"
    #metric =
    #loss = "binary_crossentropy"
    loss = f1_loss
    inputs1 = Input(shape=(width, height, 1)) 
    inputs2 = Input(shape=(width, height, 1))

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1) 
    x = Dropout(0.2)(x)
    x = MaxPooling2D((2,2), padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2), padding="same")(x)     
    x = Conv2D(32, (3,3), activation= "relu", padding="same")(x)     

    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2) 
    y = Dropout(0.2)(y)
    y = MaxPooling2D((2,2), padding="same")(y) 
    y = Conv2D(32, (3,3), activation="relu", padding="same")(y) 
    y = MaxPooling2D((2,2), padding="same")(y)     
    y = Conv2D(32, (3,3), activation= "relu", padding="same")(y)     

    z = Add()([x, y])     
    
    z = Conv2D(32, (3,3), activation= "relu", padding="same")(z)
    z = UpSampling2D((2,2))(z) 
    z = Conv2D(64, (3,3), activation="relu", padding="same")(z) 
    z = MaxPooling2D((2,2), padding="same")(z)
    z = Conv2D(128, (3,3), activation="relu", padding="same")(z) 
    z = UpSampling2D((2,2))(z) 
    z = Conv2D(32, (3,3), activation= "relu", padding="same")(z)
    z = UpSampling2D((2,2))(z)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(z) 

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]
    return model



def create_multi_input_model3(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #optimizer="sgd"
    #metric =
    #loss = "binary_crossentropy"
    loss = f1_loss
    inputs1 = Input(shape=(width, height, 1)) 
    inputs2 = Input(shape=(width, height, 1))

    x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, y]) 
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]
    return model



def create_multi_input_model4(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    #optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    optimizer = "adam"
    #optimizer="sgd"
    #metric =
    #loss = "binary_crossentropy"
    loss = f1_loss
    inputs1 = Input(shape=(width, height, 1)) 
    inputs2 = Input(shape=(width, height, 1))

    x = Conv2D(64, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(64, (3,3), activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, y]) 
    x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]
    #model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[f1, "accuracy"])
    return model



def create_multi_input_model5(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    optimizer = "adam"
    loss = f1_loss
    inputs1 = Input(shape=(width, height, 1)) 
    inputs2 = Input(shape=(width, height, 1))

    x1 = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1)
    y1 = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs1)
    y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    x = Concatenate()([x1, x_dilated, y1, y_dilated]) 
    x = MaxPooling2D((2,2), padding="same")(x)

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)

    x = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)

    x = UpSampling2D((16,16))(x)

    x_short = Conv2D(1, (3,3), activation="relu", padding="same")(x1)
    y_short = Conv2D(1, (3,3), activation="relu", padding="same")(y1)
    x_dil_short = Conv2D(1, (3,3), activation="relu", padding="same")(x_dilated)
    y_dil_short = Conv2D(1, (3,3), activation="relu", padding="same")(y_dilated)
    x = Concatenate()([x_dil_short, y_dil_short, x_short, y_short, x1, y1, x])
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1])   #"binary_crossentropy", metrics=[f1]    
    return model





def create_multi_input_model6(width, height): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    optimizer = "adam"
    loss = f1_loss
    inputs1 = Input(shape=(width, height, 1)) 
    inputs2 = Input(shape=(width, height, 1))

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs1)
    y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, x_dilated, y, y_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    #x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    #x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    #x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    return model



def create_multi_input_variant_sizes_model(): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    optimizer = "adam"
    loss = f1_loss
    inputs1 = Input(shape=(None, None, 1)) 
    inputs2 = Input(shape=(None, None, 1))

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs1)
    y = Conv2D(32, (3,3), activation="relu", padding="same")(inputs2)
    x_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs1)
    y_dilated = Conv2D(32, (3,3), dilation_rate=2, activation="relu", padding="same")(inputs2)
    x1 = Concatenate()([x, x_dilated, y, y_dilated]) 
    x1 = Conv2D(64, (1,1), activation="relu", padding="same")(x1)
    #x1 = BatchNormalization()(x1)
    x = MaxPooling2D((2,2), padding="same")(x1)

    x2 = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x2 = BatchNormalization()(x2)
    x = MaxPooling2D((2,2), padding="same")(x2)

    x3 = Conv2D(96, (3,3), activation="relu", padding="same")(x)
    x3 = Conv2D(64, (1,1), activation="relu", padding="same")(x3)
    #x3 = BatchNormalization()(x3)
    x = MaxPooling2D((2,2), padding="same")(x3)

    x4 = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x4 = Conv2D(64, (1,1), activation="relu", padding="same")(x4)
    #x4 = BatchNormalization()(x4)
    x = MaxPooling2D((2,2), padding="same")(x4)

    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (1,1), activation="relu", padding="same")(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x4]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x3]) 

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x2]) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = Concatenate()([x, x1]) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    outputs = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 


    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=[f1, "accuracy"])  
    return model






def create_classification_model(width=32, height=32): 
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.layers import UpSampling2D, BatchNormalization, Concatenate, Average
    from tensorflow.keras.models import Model
    #optimizer = "adam" 
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    #x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    #x = MaxPooling2D((2,2), padding="same")(x)
    #x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    #x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    #x = MaxPooling2D((2,2), padding="same")(x)
    #x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    #x = MaxPooling2D((2,2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (3,3), activation="relu", padding="same")(x) 
    x = Flatten()(x)
    x = Dense(10, activation="relu")(x)
    outputs = Dense(4, activation="softmax")(x) 
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])#
    return model


def transfer_learning_fashion_mnist_model(): 
    # loads a model with non-trainable layers from a pretrained fashion-mnist classifier.
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
    from tensorflow.keras.models import Model

    width = 32
    
    inputs = Input(shape=(width, width, 1))
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="1", trainable=False)(inputs) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="2", trainable=False)(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="3", trainable=False)(x) 
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="4", trainable=False)(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="5", trainable=False)(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="6", trainable=False)(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same", name="7", trainable=False)(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same", name="8", trainable=False)(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same", name="9", trainable=False)(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same", name="13")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same", name="14")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same", name="15")(x)

    x = Flatten()(x) 
    x = Dense(128, activation='relu', name="10")(x) 
    x = Dropout(0.2, name="11")(x) 
    outputs = Dense(4, activation='softmax', name="12")(x)

    model2 = Model(inputs=inputs, outputs=outputs)
    model2.compile(optimizer="adam",
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    model2.load_weights("fashion_mnist_checkpoint_epoch_00100.h5", by_name=True)
    #model2.summary()
    return model2
    


def autoencoder(width): 
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
    from tensorflow.keras.models import Model 

    inputs = Input(shape=(width, width, 1)) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2))(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2))(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = MaxPooling2D((2,2))(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 

    x = UpSampling2D((2,2))(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = UpSampling2D((2,2))(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same")(x) 

    outputs = Conv2D(1, (3,3), activation="linear", padding="same")(x) 

    model = Model(inputs=inputs, outputs=outputs) 
    model.compile(optimizer="adam", loss="mse") 
    return model


def autoencoder_transfer_model(): 
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
    from tensorflow.keras.models import Model

    autoenc = autoencoder(32) 
    autoenc.load_weights("checkpoints/autoencoder_cp_epoch_00100.h5") 
    autoenc.trainable = False

    inputs = Input(shape=(32, 32, 1)) 
    x = autoenc(inputs) 
    x = Flatten()(x) 
    x = Dense(128, activation="relu")(x) 
    x = Dense(8, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x) 

    model = Model(inputs=inputs, outputs=output) 
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model






def print_summaries(): 
    from tensorflow.keras.backend import clear_session
    print("Single Input Segmentation: ")
    model = create_multiscale_segmentation_model3(64, 64) 
    model.summary() 
    clear_session() 
    print("Multi Input Segmentation:")
    model = create_multi_input_model6(64, 64)
    model.summary() 
    clear_session()  
    print("Multifilter Segmentation:")
    model = create_multifilter_segmentation_model5(64, 64)
    model.summary() 
    clear_session()  
    print("Multifilter Detection:")
    model = create_multifilter_detection_model3(64, 64)
    model.summary() 
    clear_session() 

if __name__ == "__main__": 
    print_summaries()