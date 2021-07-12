import numpy as np 
from one_class_classification import convert_to_array_2d
from sklearn.model_selection import train_test_split
import tensorflow as tf 


window_list = np.load("full_raw_gm_train_data_5_to_30-31320.npy")

R = [] 
Y = []
for w in window_list: 
    r = np.mean(w.shape) 
    R.append(r) 
    #Y.append([1, r])
X = convert_to_array_2d(window_list, 16, 16) 
Y = np.ones(len(X))
R = np.asarray(R) 
Y = np.asanyarray(Y)
print(X.shape, Y.shape, R.shape) 



window_list = np.load("ngm_train_data_norway.npy")

X_ngm = convert_to_array_2d(window_list, 16, 16) 
#Y_ngm = np.zeros((len(X_ngm), 2))
Y_ngm = np.zeros(len(X_ngm)) 
R_ngm = np.zeros(len(X_ngm))

print(X_ngm.shape, Y_ngm.shape, R_ngm.shape)


X = np.concatenate((X, X_ngm), axis=0)  
Y = np.concatenate((Y, Y_ngm), axis=0)
R = np.concatenate((R, R_ngm), axis=0)

print(X.shape, Y.shape, R.shape)

Y_new = np.zeros((len(Y), 2)) 
for i in range(len(Y)): 
    Y_new[i][0] = Y[i] 
    Y_new[i][1] = R[i] 
Y = Y_new   
print(X.shape, Y.shape)


X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)


y_train_new = [y[0] for y in y_train]
y_val_new = [y[0] for y in y_val]
y_test_new = [y[0] for y in y_test]

r_train = np.asarray([y[1] for y in y_train])
r_val = np.asarray([y[1] for y in y_val])
r_test = np.asarray([y[1] for y in y_test])


y_train = tf.keras.utils.to_categorical(np.asarray(y_train_new))
y_val = tf.keras.utils.to_categorical(np.asarray(y_val_new))
y_test = tf.keras.utils.to_categorical(np.asarray(y_test_new))

"""
for i in range(len(y_train)):
    print(y_train[i], r_train[i])
"""

def create_regression_model(width, height): 
    inputs = tf.keras.layers.Input(shape=(width, height, 1)) 

    x = tf.keras.layers.Conv2D(16, kernel_size=(3), activation="relu")(inputs) 
    x = tf.keras.layers.Conv2D(16, kernel_size=(3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=(3), activation="relu")(x)     
    x = tf.keras.layers.Conv2D(4, kernel_size=(3), activation="relu")(x) 

    x = tf.keras.layers.Flatten()(x) 
    x = tf.keras.layers.Dense(64, activation="relu")(x) 
    x = tf.keras.layers.Dense(32, activation="relu")(x) 
    x = tf.keras.layers.Dense(16, activation="relu")(x) 

    outputs = tf.keras.layers.Dense(2, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs) 
    model.compile(optimizer="adam", loss="mse") #"sgd"   "categorical_crossentropy", metrics=["mse"]
    model.summary()
    return model


def create_multioutput_model(width, height): 
    from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense 
    from tensorflow.keras.models import Model
    inputs = Input(shape=(width, height, 1)) 

    x = Conv2D(16, kernel_size=(3), activation="relu")(inputs) 
    x = Conv2D(16, kernel_size=(3), activation="relu")(x)
    x = Conv2D(8, kernel_size=(3), activation="relu")(x)     
    x = Conv2D(4, kernel_size=(3), activation="relu")(x) 

    c = Flatten()(x) 
    c = Dense(64, activation="relu")(c) 
    c = Dense(32, activation="relu")(c) 
    c = Dense(16, activation="relu")(c) 
    c = Dense(2, activation="softmax", name="c_out")(c)
    
    r = Flatten()(x) 
    r = Dense(64, activation="relu")(r) 
    r = Dense(32, activation="relu")(r) 
    r = Dense(16, activation="relu")(r) 
    r = Dense(1, activation="linear", name="r_out")(r)

    model = Model(inputs=inputs, outputs=[c, r]) 
    model.compile(optimizer="adam", loss=["categorical_crossentropy", "mse"], metrics=["accuracy"])
    model.summary()
    return model


model = create_multioutput_model(16, 16)
epochs = 50
batch_size = 128
model.fit(  x_train, [y_train, r_train], 
            epochs=epochs, 
            batch_size=batch_size,                
            validation_data=[x_val, [y_val, r_val]]
         )    

score = model.evaluate(x_test, [y_test, r_test])
print("Score: ", score)

"""
model = create_regression_model(16, 16)
epochs = 500
batch_size = 128
model.fit(  x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size,                
            validation_data=[x_val, y_val]
         )

score = model.evaluate(x_test, y_test)
print("Score: ", score)

"""