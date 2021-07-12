import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2
import sys
import preprocessing as pp


def fashion_mnist_model(width): 
    

    inputs = Input(shape=(width, width, 1))
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="1")(inputs) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="2")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="3")(x) 
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="4")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="5")(x) 
    x = Conv2D(32, (3,3), activation="relu", padding="same", name="6")(x) 

    x = Conv2D(64, (3,3), activation="relu", padding="same", name="7")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same", name="8")(x) 
    x = Conv2D(64, (3,3), activation="relu", padding="same", name="9")(x) 

    x = Flatten()(x) 
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.2)(x) 
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

def test_transfer_learning(): 
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

    x = Flatten()(x) 
    x = Dense(128, activation='relu', name="10")(x) 
    x = Dropout(0.2, name="11")(x) 
    outputs = Dense(4, activation='softmax', name="12")(x)

    model2 = Model(inputs=inputs, outputs=outputs)
    model2.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


    model2.load_weights("fashion_mnist_checkpoint_epoch_00100.h5", by_name=True)
    model2.summary()

    """    
    sys.exit()
    model = fashion_mnist_model(32)
    
    inputs = Input(shape=(32, 32, 1))
    x = model(inputs) #.layers[-5].output
    #model.summary()
     
    #model.layers = model.layers[:-3]
    #model.layers = model.layers.pop() 
    #model.layers.pop()
    #model.layers.pop()
    #model.layers.pop()
    out = Dense(4, activation="softmax")(x)
    model2 = Model(inputs=inputs, outputs=out) 
    model2.compile( optimizer="adam",
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    model2.summary()
    """





def transfer_learning(): 
    import h5py 
    from models import transfer_learning_fashion_mnist_model
    import preprocessing as pp
    from sklearn.model_selection import train_test_split

    dataset = h5py.File("classification_dataset_32by32_augmented.h5", "r") 
    x_train = dataset.get("x_train")
    x_train = np.array(x_train) 
    y_train = dataset.get("y_train")
    y_train = np.array(y_train)
    x_test = dataset.get("x_test") 
    x_test = np.array(x_test) 
    y_test = dataset.get("y_test") 
    y_test = np.array(y_test)
    dataset.close() 

    method = "laplacian"
    #method = "slope"
    pp.transform_dataset_list([x_train, x_test], method)
    
    for x in range(len(x_train)): 
        x_train[x] -= np.min(x_train[x]) 
        x_train[x] /= np.max(x_train[x]) 
    for x in range(len(x_test)): 
        x_test[x] -= np.min(x_test[x]) 
        x_test[x] /= np.max(x_test[x])  
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)  
    
    #x_train, y_train = pp.rotate_classification_dataset(x_train, y_train)
    #print(len(x_train), len(y_train))
    #x_train = np.asarray(x_train)
    #y_train = np.asarray(y_train)
    #x_train = np.concatenate((x_train, np.flip(x_train, axis=1)))
    #y_train = np.concatenate((y_train, np.flip(y_train, axis=1)))
    
    x_train = x_train.reshape(-1, 32, 32, 1)
    x_val = x_val.reshape(-1, 32, 32, 1)
    x_test = x_test.reshape(-1, 32, 32, 1)
    print(y_train.shape)
    model = transfer_learning_fashion_mnist_model() 
    model.summary() 
    epochs = 50
    batch_size = 32
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=[x_val, y_val])
    score = model.evaluate(x_test, y_test) 
    print(score)











#transfer_learning() 
#sys.exit()








mnist = tf.keras.datasets.fashion_mnist
#mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
method = "laplacian"  
x_train = x_train.copy()
x_test = x_test.copy()
pp.transform_dataset_list([x_train, x_test], method)
"""
for x in range(len(x_train)):
    #print(type(x_train[x]))
    x_train[x] -= np.min(x_train[x]) 
    x_train[x] = x_train[x] / np.max(x_train[x]) 
for x in range(len(x_test)): 
    x_test[x] -= np.min(x_test[x]) 
    x_test[x] = x_test[x] / np.max(x_test[x])  
""" 
#
#rmsprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
width = 32

X = [] 
for x in x_train: 
    X.append(cv2.resize(x, (width, width)))
Y = [] 
for x in x_test: 
    Y.append(cv2.resize(x, (width, width)))

x_train = np.asarray(X) 
x_test = np.asarray(Y)


x_train = x_train.reshape((-1, width, width, 1))
x_test = x_test.reshape((-1, width, width, 1))

model = fashion_mnist_model(32)
model.summary()
#model.load_weights("fashion_mnist_checkpoint_epoch_00160.h5") 

callbacks = [] 
cp_callback = ModelCheckpoint("fashion_mnist_checkpoint_epoch_{epoch:05d}.h5", verbose=1, save_weights_only=True, period=20)
callbacks.append(cp_callback)

model.fit(x_train, y_train, epochs=100, callbacks=callbacks)

score = model.evaluate(x_test, y_test)
print(score)
#model.save("fashion_mnist_model.h5")