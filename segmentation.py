import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import create_conv_segmentation_model6 
from DTM_frame_data_test import get_DTM_frame_data 
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from augment_by_smallsizing import smallsize_matrix_general
from utils import get_window_by_position


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
    model = create_conv_segmentation_model6(width, height)
    model.load_weights("checkpoints/segmentation_model6_sobel_epoch_00200.h5")
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

    plt.subplot(1,2,1) 
    plt.imshow(matrix, cmap="Greys") 
    plt.subplot(1,2,2) 
    plt.imshow(label_matrix, cmap="gist_rainbow") 
    plt.show()
    np.save("images/prediction_frame_vieritz_64kernel.npy", label_matrix)





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

    for i in range(len(x_train)): 
        p, q, r, s, t = pp.get_gradients(x_train[i])
        #x_train[i] = pp.slope(p, q)
        #x_train[i] = pp.unsphericity_curvature(p, q, r, s, t)
        #x_train[i] = pp.cv2_blur(x_train[i], (41, 41))
        #x_train[i] = pp.vertical_curvature(p, q, r, s, t)
        x_train[i] = pp.cv2_sobel2D(x_train[i])
        #x_train[i] = pp.slrm_cv2_average(x_train[i], (40, 40))



    for i in range(len(x_test)): 
        p, q, r, s, t = pp.get_gradients(x_test[i])
        #x_test[i] = pp.slope(p, q)
        #x_test[i] = pp.unsphericity_curvature(p, q, r, s, t)
        #x_test[i] = pp.cv2_blur(x_test[i], (41, 41))
        #x_test[i] = pp.vertical_curvature(p, q, r, s, t)
        x_test[i] = pp.cv2_sobel2D(x_test[i])
        #x_test[i] = pp.slrm_cv2_average(x_test[i], (40, 40))






    x_train, y_train = pp.rotate_dataset(x_train, y_train)

    x_train = np.concatenate((x_train, np.flip(x_train, axis=2)))
    y_train = np.concatenate((y_train, np.flip(y_train, axis=2)))

    x_train, y_train = shuffle(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3) 

    x_train = np.asarray(x_train).reshape((-1, 64, 64, 1)) 
    y_train = np.asarray(y_train).reshape((-1, 64, 64, 1)) 
    x_val = np.asarray(x_val).reshape((-1, 64, 64, 1)) 
    y_val = np.asarray(y_val).reshape((-1, 64, 64, 1)) 
    x_test = np.asarray(x_test).reshape((-1, 64, 64, 1)) 
    y_test = np.asarray(y_test).reshape((-1, 64, 64, 1))
    """
    x_train = np.clip(x_train, -0.2, 0.2)
    x_val = np.clip(x_val, -0.2, 0.2)
    x_test = np.clip(x_test, -0.2, 0.2) 
    """
    x_train = np.clip(x_train, 0, 400)
    x_val = np.clip(x_val, 0, 400)
    x_test = np.clip(x_test, 0, 400) 

    for i in [x_train, y_train, x_val, y_val, x_test, y_test]: 
        print(i.shape)

    width = x_train[0].shape[0]
    height = x_train[0].shape[1]
    model = create_conv_segmentation_model6(width, height)
    model.summary()
    epochs = 200
    batch_size = 128   #96


    checkpoint_path = "checkpoints/segmentation_model6_sobel_epoch_{epoch:05d}.h5"
    checkpoint_frequency = 100
    callbacks = []    
        
    cp_callback = tf.keras.callbacks.ModelCheckpoint( checkpoint_path, 
                                                        verbose=1, 
                                                        save_weights_only=True,
                                                        period=checkpoint_frequency )
    callbacks.append(cp_callback)

    model.fit(  x_train, 
                y_train, 
                epochs=epochs, 
                batch_size=batch_size,  
                shuffle=True,
                callbacks=callbacks,              
                validation_data=[x_val, y_val]  )

    score = model.evaluate(x_test, y_test)
    print("Score: ", score)
    return model


#predict_frames_vieritz(train(), 64, 64)
predict_frames_vieritz_by_saved_weights(64, 64)

