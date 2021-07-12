import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import *
from DTM_frame_data_test import get_DTM_frame_data 
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from augment_by_smallsizing import smallsize_matrix_general
from utils import get_window_by_position
import sys

def predict_frames_vieritz(model, width, height):
    m = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(m, 2)[0] 
    p, q, r, s, t = pp.get_gradients(matrix)
    #matrix = pp.slope(p, q)
    #matrix = pp.unsphericity_curvature(p, q, r, s, t)  ###
    #matrix = pp.cv2_blur(matrix, (41, 41))
    #matrix = pp.vertical_curvature(p, q, r, s, t)
    #matrix = pp.slrm_cv2_average(matrix, (40, 40))
    #matrix = np.clip(matrix, -0.2, 0.2)
    matrix = pp.cv2_sobel2D(matrix)                    ###
    matrix = np.clip(matrix, 0, 400)
    print(matrix.shape)
    #plt.imshow(matrix, cmap="Greys")
    #plt.show()
    label_matrix = np.zeros(matrix.shape)
    for i in range(0, matrix.shape[0] - width + 1, 12): 
        for j in range(0, matrix.shape[1] - height + 1, 12): 
            #print(i, j)
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            window = window_and_box[0] 
            window = np.reshape(window, (1, width, height, 1))
            window -= np.min(window)
            pred = model.predict(window)
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    label_matrix[i + x][j + y] = max(label_matrix[i + x][j + y], pred[x][y])  
                    #label_matrix[i + x][j + y] = (label_matrix[i + x][j + y] + pred[x][y]) / 2            

    plt.subplot(2,1,1) 
    plt.imshow(matrix, cmap="Greys") 
    plt.subplot(2,1,2) 
    plt.imshow(label_matrix, cmap="gist_rainbow") 
    plt.show()
    np.save("images/prediction_frame_vieritz_64kernel.npy", label_matrix)



def predict_frames_vieritz_by_saved_weights(width, height):
    model = create_multi_input_model2(width, height)
    model.load_weights("checkpoints/multi_input_segmentation_model2_sobel_usc_dice_epoch_00300.h5")
    m = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(m, 2)[0] 
    p, q, r, s, t = pp.get_gradients(matrix)

    matrix1 = pp.cv2_sobel2D(matrix)
    matrix2 = pp.unsphericity_curvature(p, q, r, s, t)  
    #matrix1 = pp.slope(p, q)
    #matrix2 = pp.aspect(p, q)
    #matrix2 = np.deg2rad(matrix2)

    label_matrix = np.zeros(matrix.shape)
    for i in range(0, matrix.shape[0] - width + 1, 12): 
        for j in range(0, matrix.shape[1] - height + 1, 12): 
            #print(i, j)
            window_and_box1 = get_window_by_position(width, height, matrix1, i, j) 
            window_and_box2 = get_window_by_position(width, height, matrix2, i, j)
            window1 = window_and_box1[0] 
            window2 = window_and_box2[0] 
            window1 = np.reshape(window1, (1, width, height, 1))
            window2 = np.reshape(window2, (1, width, height, 1))
            #window -= np.min(window)
            pred = model.predict([window1, window2])
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    label_matrix[i + x][j + y] = max(label_matrix[i + x][j + y], pred[x][y])  
                    #label_matrix[i + x][j + y] = (label_matrix[i + x][j + y] + pred[x][y]) / 2            

    plt.subplot(2,2,1) 
    plt.imshow(matrix1, cmap="Greys") 
    plt.subplot(2,2,2) 
    plt.imshow(matrix2, cmap="Greys") # cmap=,  "gist_rainbow"
    plt.subplot(2,2,3) 
    plt.imshow(label_matrix, cmap="Greys") #    
    plt.show()
    np.save("images/prediction_frame_vieritz_64kernel_multi_input2_sobel_usc_dice.npy", label_matrix)


def reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test):
    x_train = x_train.reshape((-1, 64, 64, 1)) 
    y_train = y_train.reshape((-1, 64, 64, 1)) 
    x_val = x_val.reshape((-1, 64, 64, 1)) 
    y_val = y_val.reshape((-1, 64, 64, 1)) 
    x_test = x_test.reshape((-1, 64, 64, 1)) 
    y_test = y_test.reshape((-1, 64, 64, 1))
    return x_train, y_train, x_val, y_val, x_test, y_test


def asarray_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train) 
    x_val = np.asarray(x_val) 
    y_val = np.asarray(y_val) 
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test









def train():


    #[[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    data = get_DTM_frame_data()

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 




    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))


    # Preprocessing 

    x_train, y_train = pp.rotate_dataset(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)

    #x_train = np.concatenate((x_train, np.flip(x_train, axis=2)))
    #y_train = np.concatenate((y_train, np.flip(y_train, axis=2)))

    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3) 

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train) 
    x_val = np.asarray(x_val) 
    y_val = np.asarray(y_val) 
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_train2 = np.copy(x_train)
    x_val2 = np.copy(x_val) 
    x_test2 = np.copy(x_test) 

    
    for i in [x_train, x_val, x_test]:    
        for j in range(len(i)):           
            i[j] = pp.cv2_sobel2D(i[j])
            #p, q, r, s, t = pp.get_gradients(i[j])
            #i[j] = pp.slope(p, q)

    for i in [x_train2, x_val2, x_test2]:    
        for j in range(len(i)): 
            p, q, r, s, t = pp.get_gradients(i[j])
            i[j] = pp.unsphericity_curvature(p, q, r, s, t)     
            #i[j] = pp.aspect(p, q) 
            #i[j] = np.deg2rad(i[j])  

    x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)
    x_train2 = x_train2.reshape((-1, 64, 64, 1))
    x_val2 = x_val2.reshape((-1, 64, 64, 1)) 
    x_test2 = x_test2.reshape((-1, 64, 64, 1)) 

    """
    x_train = np.clip(x_train, -0.2, 0.2)
    x_val = np.clip(x_val, -0.2, 0.2)
    x_test = np.clip(x_test, -0.2, 0.2) 
    """
    """
    x_train = np.clip(x_train, 0, 400)
    x_val = np.clip(x_val, 0, 400)
    x_test = np.clip(x_test, 0, 400) 
    """

    #for i in [x_train, y_train, x_val, y_val, x_test, y_test]: 
    #    print(i.shape)

    width = x_train[0].shape[0]
    height = x_train[0].shape[1]
    model = create_multi_input_model2(width, height)
    model.summary()
    epochs = 10
    batch_size = 96 #128


    checkpoint_path = "checkpoints/multi_input_segmentation_model2_sobel_usc_dice_epoch_{epoch:05d}.h5"
    checkpoint_frequency = 100
    callbacks = []    
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint( checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=checkpoint_frequency )
    callbacks.append(cp_callback)

    model.fit(  [x_train, x_train2],  # {"inputs1":x_train, "inputs2":x_train2}
                y_train, 
                epochs=epochs, 
                batch_size=batch_size,  
                shuffle=False,
                callbacks=callbacks,              
                validation_data=[[x_val, x_val2], y_val]  )

    score = model.evaluate([x_test, x_test2], y_test)
    print("Score: ", score)
    return model



def train_slope_laplacian(mirror=False):


    #[[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    data = get_DTM_frame_data()

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 




    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))


    # Preprocessing 

    x_train, y_train = pp.rotate_dataset(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    if mirror == True:
        x_train = np.flip(x_train, axis=2)
        y_train = np.flip(y_train, axis=2)

    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3) 

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train) 
    x_val = np.asarray(x_val) 
    y_val = np.asarray(y_val) 
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_train2 = np.copy(x_train)
    x_val2 = np.copy(x_val) 
    x_test2 = np.copy(x_test) 

    
    for i in [x_train, x_val, x_test]:    
        for j in range(len(i)):           
            #i[j] = pp.cv2_sobel2D(i[j])
            p, q, r, s, t = pp.get_gradients(i[j])
            i[j] = pp.slope(p, q)

    for i in [x_train2, x_val2, x_test2]:    
        for j in range(len(i)): 
            p, q, r, s, t = pp.get_gradients(i[j])
            #i[j] = pp.unsphericity_curvature(p, q, r, s, t)     
            #i[j] = pp.aspect(p, q) 
            #i[j] = np.deg2rad(i[j])  
            i[j] = pp.laplacian(r, t)

    x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)
    x_train2 = x_train2.reshape((-1, 64, 64, 1))
    x_val2 = x_val2.reshape((-1, 64, 64, 1)) 
    x_test2 = x_test2.reshape((-1, 64, 64, 1)) 

    """
    x_train = np.clip(x_train, -0.2, 0.2)
    x_val = np.clip(x_val, -0.2, 0.2)
    x_test = np.clip(x_test, -0.2, 0.2) 
    """
    """
    x_train = np.clip(x_train, 0, 400)
    x_val = np.clip(x_val, 0, 400)
    x_test = np.clip(x_test, 0, 400) 
    """

    #for i in [x_train, y_train, x_val, y_val, x_test, y_test]: 
    #    print(i.shape)

    width = x_train[0].shape[0]
    height = x_train[0].shape[1]
    model = create_multi_input_model2(width, height)
    model.summary()
    epochs = 1000
    batch_size = 96 #128


    checkpoint_path = "checkpoints/multi_input_segmentation_model2_slope_laplacian_dice_epoch_{epoch:05d}.h5"
    checkpoint_frequency = 100
    callbacks = []    
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint( checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=checkpoint_frequency )
    callbacks.append(cp_callback)

    model.fit(  [x_train, x_train2],  # {"inputs1":x_train, "inputs2":x_train2}
                y_train, 
                epochs=epochs, 
                batch_size=batch_size,  
                shuffle=False,
                callbacks=callbacks,              
                validation_data=[[x_val, x_val2], y_val]  )

    score = model.evaluate([x_test, x_test2], y_test)
    print("Score: ", score)
    return model




#predict_frames_vieritz(train(), 64, 64)
#predict_frames_vieritz_by_saved_weights(64, 64)
#train()
train_slope_laplacian()

