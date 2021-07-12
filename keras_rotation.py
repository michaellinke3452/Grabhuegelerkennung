# source until line 22: https://qiita.com/li-li-qiita/items/5e5c71d0ce2bb7922260
# date: 02.03.2021

import numpy as np
#from skimage.data import camera
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
from custom_layers import *
import sys
"""
# Create a multi dimensional image
image =  camera().astype(np.float32)
image = np.expand_dims(np.expand_dims(image, axis=2), axis=0)
image = np.concatenate((image, 0.5*image), 0)
image = np.concatenate((image, 0.5*image), 0)
image3D = np.expand_dims(image, axis=0)
image3D = np.concatenate((image3D, 0.5*image3D), 0)

image_tsp = K.permute_dimensions(image3D, (0, 1, 3, 2, 4))
image_cw90 = K.reverse(image_tsp, axes=-2)  # clock wise
image_ccw90 = K.reverse(image_tsp, axes=-3)  # counter clock wise
"""

image = np.load("images/vieritz.npy") 
#image =  image.astype(np.float32)
plt.imshow(image, cmap="gray") 
plt.show()
width, height = image.shape
"""
import sys
image2 = image.reshape((width, height, 1)) 
image2 = image2.reshape((width, height)) 
plt.imshow(image2, cmap="gray") 
plt.show()
"""
#sys.exit()


image_stack = [image, image, image] 
image_stack = np.asarray(image_stack) 
image_stack = np.reshape(image_stack, (-1, width, height, 1)) 
"""
for i in image_stack:
    #image = tf.reshape(image, (width, height, 1))
    image = tf.reshape(i, (width, height))
    image_tsp = K.transpose(image)
    image_tsp = Gradient(axis=1)(image_tsp) 
    #image_tsp = K.permute_dimensions(image, (1,0))
    #image_tsp = K.permute_dimensions(image, (0, 2, 1))
    image_cw90 = K.reverse(image_tsp, axes=0)  # clock wise
    image_ccw90 = K.reverse(image_tsp, axes=1)  # counter clock wise
    # end of works
"""
#ist = tf.Variable(image_stack)
#y = Gradient(axis=0)(image_stack)
y = Laplacian()(image_stack)
y = Rotate()(y)

#y = Gradient(axis=1)(y)

with  tf.Session() as sess:
    x = sess.run(y)
    #x = x.reshape((width, height)) 
    for i in x:
        print(i.shape)
        i = np.clip(i, -0.1, 0.1)
        
        plt.imshow(i.reshape((i.shape[0], i.shape[1])), cmap="Greys")
        #plt.imshow(i.reshape((height, width)), cmap="gray") 
        plt.show()


sys.exit()

"""
image = np.reshape(image, (-1, width, height, 1))
print(type(image))
image_tsp = K.permute_dimensions(image, (0, 2, 3, 1))
image_tsp = image
print(type(image_tsp))
image_cw90 = K.reverse(image_tsp, axes=-3)  # clock wise
print(type(image_cw90))
"""
"""
image = np.reshape(image, (width, height, 1))
#image = np.reshape(image, (width, height))
#image_tsp = K.transpose(image)
image = Gradient(axis=1)(image) 
image_tsp = K.permute_dimensions(image, (2, 0, 1))
image_tsp = K.permute_dimensions(image, (0, 2, 1))
image_cw90 = K.reverse(image_tsp, axes=0)  # clock wise
image_ccw90 = K.reverse(image_tsp, axes=1)  # counter clock wise

"""
# works: 
print(image.shape)
image = tf.reshape(image, (width, height, 1))
image = tf.reshape(image, (width, height))
image_tsp = K.transpose(image)
image_tsp = Gradient(axis=1)(image_tsp) 
#image_tsp = K.permute_dimensions(image, (1,0))
#image_tsp = K.permute_dimensions(image, (0, 2, 1))
image_cw90 = K.reverse(image_tsp, axes=0)  # clock wise
image_ccw90 = K.reverse(image_tsp, axes=1)  # counter clock wise
# end of works



with  tf.Session() as sess:
    x = sess.run(image_cw90)
    #x = x.reshape((width, height)) 
    x = np.clip(x, -0.2, 0.2)
    plt.imshow(x, cmap="gray") 
    plt.show()

"""  
import itertools
l = list(itertools.permutations([0, 1, 2, 3]))
print(l)
image = np.reshape(image, (width, height, 1)) 
im = [image]
image = np.asarray(im)
print(image.shape)
#
#inputs = tf.Variable(image)
#y = tf.roll(inputs, 1, self.axis) - inputs
#y = tf.roll(image, 1, 1) - image



image = Gradient(axis=2)(image) 



#image = K.permute_dimensions(image, (3, 0, 1, 2))
for perm in l:
    imagep = K.permute_dimensions(image, perm)
    for perm2 in l:
        print(perm, perm2)
        imagep2 = K.permute_dimensions(imagep, perm2)
        #y = Reflectance()(image)
        y = Rotate()(imagep2)
        with tf.Session() as sess:
            c = sess.run(y) 
            c = c.reshape((width, height)) 
            c = np.clip(c, -0.2, 0.2)

            plt.imshow(c, cmap="gray") 
            plt.show()
"""