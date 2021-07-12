import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import *
#from DTM_frame_data_test import get_DTM_frame_data 
import preprocessing as pp
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
from augment_by_smallsizing import smallsize_matrix_general
from utils import get_window_by_position
import sys


def predict_frames_by_saved_weights_rotated(npyname, weights, method, width=64, height=64, smallsize=0, model=None):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session

    print(npyname)
    if model == None:
        model = create_multiscale_segmentation_model(width, height)
    model.load_weights(weights)
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    """
    matrix = smallsize_matrix_general(m, 2)[0] 
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)
    """
    if smallsize == 1: 
        matrix = smallsize_matrix_general(m, 2)[0] 
    else: 
        matrix = m
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)

    data_list = [[matrix0, matrix1, matrix2, matrix3]]
    data_list = pp.transform_dataset_list(data_list, method)
    matrix0 = data_list[0][0]
    matrix1 = data_list[0][1] 
    matrix2 = data_list[0][2] 
    matrix3 = data_list[0][3]
    #plt.imshow(matrix0) 
    #plt.show()
    predictions = []
    m_index = 0
    
    for matrix in [matrix0, matrix1, matrix2, matrix3]: 
        m_index += 1
        label_matrix = np.zeros(matrix.shape)
        X = [x for x in range(0, matrix.shape[0] - width + 1, 60)]
        last = matrix.shape[0] - width 
        if X[-1] < last: 
            X.append(last)
        Y = [y for y in range(0, matrix.shape[1] - height +1, 60)]
        last = matrix.shape[1] - height 
        if Y[-1] < last: 
            Y.append(last)
        #print(X[-1], Y[-1])
        #print(X[-1] + width, Y[-1] + height)
        #print(matrix.shape[0], matrix.shape[1])

        for i in X: 
            for j in Y: 
                #print(i, j)
                print("matrix ", m_index, ": ", i, " of ", matrix.shape[0], " ",  j, " of ", matrix.shape[1], "            ", end="\r")
                window_and_box = get_window_by_position(width, height, matrix, i, j) 
                window = window_and_box[0] 
                window = np.reshape(window, (1, width, height, 1))
                pred = model.predict(window)
                pred = np.reshape(pred, (width, height))
                for x in range(pred.shape[0]): 
                    for y in range(pred.shape[1]): 
                        p = max(label_matrix[i + x][j + y], pred[x][y]) 
                        #if p >= 0.99: 
                        label_matrix[i + x][j + y] = p 
        predictions.append(label_matrix) 
         
    label_matrix = np.rot90(np.zeros(predictions[0].shape), k=1) 
    for i in range(0, 4): 
        predictions[i] = np.rot90(predictions[i], k=3-i)  
        label_matrix = label_matrix + predictions[i] 
        
    label_matrix /= 4
    label_matrix = np.rot90(label_matrix, k=1)
    clear_session()
    print()
    return label_matrix




def predict_multifilter_segmentation(npyname, model, width=64, height=64, smallsize=0, verbose=True):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session
    from preprocessing import transform

    print(npyname)  
    
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    
    if smallsize > 1: 
        matrix = smallsize_matrix_general(m, smallsize)[0] 
    else: 
        matrix = m
       
    predictions = []
    m_index = 0
    
     
    m_index += 1
    label_matrix = np.zeros(matrix.shape)
    X = [x for x in range(0, matrix.shape[0] - width + 1, 60)]
    last = matrix.shape[0] - width 
    if X[-1] < last: 
        X.append(last)
    Y = [y for y in range(0, matrix.shape[1] - height +1, 60)]
    last = matrix.shape[1] - height 
    if Y[-1] < last: 
        Y.append(last)

    for i in X: 
        for j in Y: 
            print("matrix ", m_index, ": ", i, " of ", matrix.shape[0], " ",  j, " of ", matrix.shape[1], "            ", end="\r")
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            window = window_and_box[0] 
            window = np.reshape(window, (1, width, height, 1))
            pred = model.predict(window)
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    p = max(label_matrix[i + x][j + y], pred[x][y]) 
                    label_matrix[i + x][j + y] = p        
    
    
    #clear_session()
    if verbose == True:
        print()
        m = transform(m, "laplacian") #"pseudo_slope"
        m = np.clip(m, -1, 1)
        plt.subplot(1,2,1) 
        plt.imshow(m, cmap="Greys") #
        plt.subplot(1,2,2) 
        plt.imshow(label_matrix, cmap="hot") # label_matrix_class   "Greys"  
        plt.show()
    return label_matrix




def predict_multifilter_segmentation_rotated(npyname, model, width=64, height=64, smallsize=0, verbose=True):
    from utils import get_four_rotation_matrices, remove_outliers_from_matrix
    from tensorflow.keras.backend import clear_session
    from preprocessing import transform

    print(npyname)  
    
    m = np.load(npyname)  
    
    if len(m.shape) > 2: 
        m = m[0]  
    #m = remove_outliers_from_matrix(m, 10, mean_type="median") 
    if smallsize > 1: 
        matrix = smallsize_matrix_general(m, smallsize)[0] 
    else: 
        matrix = m

    
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)

    data_list = [[matrix0, matrix1, matrix2, matrix3]]

    matrix0 = data_list[0][0]
    matrix1 = data_list[0][1] 
    matrix2 = data_list[0][2] 
    matrix3 = data_list[0][3]
    
    predictions = []
    m_index = 0
    
    for matrix in [matrix0, matrix1, matrix2, matrix3]: 
        m_index += 1
        label_matrix = np.zeros(matrix.shape)
        X = [x for x in range(0, matrix.shape[0] - width + 1, 60)]
        last = matrix.shape[0] - width 
        if X[-1] < last: 
            X.append(last)
        Y = [y for y in range(0, matrix.shape[1] - height +1, 60)]
        last = matrix.shape[1] - height 
        if Y[-1] < last: 
            Y.append(last)

        for i in X: 
            for j in Y: 
                print("matrix ", m_index, ": ", i, " of ", matrix.shape[0], " ",  j, " of ", matrix.shape[1], "            ", end="\r")
                window_and_box = get_window_by_position(width, height, matrix, i, j) 
                window = window_and_box[0] 
                window = np.reshape(window, (1, width, height, 1))
                pred = model.predict(window)
                pred = np.reshape(pred, (width, height))
                for x in range(pred.shape[0]): 
                    for y in range(pred.shape[1]): 
                        p = max(label_matrix[i + x][j + y], pred[x][y]) 
                        label_matrix[i + x][j + y] = p 
        predictions.append(label_matrix) 
         
    label_matrix = np.rot90(np.zeros(predictions[0].shape), k=1) 
    for i in range(0, 4): 
        predictions[i] = np.rot90(predictions[i], k=3-i)  
        label_matrix = label_matrix + predictions[i] 
        
    label_matrix /= 4
    label_matrix = np.rot90(label_matrix, k=1)
    #clear_session()
    if verbose == True:
        print()
        plt.subplot(1,2,1) 
        plt.imshow(transform(m, "slope"), cmap="Greys") 
        plt.subplot(1,2,2) 
        plt.imshow(label_matrix, cmap="hot") # label_matrix_class   "Greys"  
        plt.show()
    return label_matrix





def predict_multi_input_frames(npyname, weights, methods, width=64, height=64, smallsize=0, model=None, resolution=2):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session
    print(npyname)
    if model == None:
        model = create_multiscale_segmentation_model(width, height)
    model.load_weights(weights)
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    """
    matrix = smallsize_matrix_general(m, 2)[0] 
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)
    """
    if smallsize == 1: 
        matrix = smallsize_matrix_general(m, 2)[0] 
    else: 
        matrix = m
    #matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)

    #data_list = [[matrix0, matrix1, matrix2, matrix3]]
    #data_list = pp.transform_dataset_list(data_list, method)
    #matrix0 = data_list[0][0]
    #matrix1 = data_list[0][1] 
    #matrix2 = data_list[0][2] 
    #matrix3 = data_list[0][3]
    #plt.imshow(matrix0) 
    #plt.show()
    #predictions = []    
    
    label_matrix = np.zeros(matrix.shape)
    X = [x for x in range(0, matrix.shape[0] - width + 1, 60)]
    last = matrix.shape[0] - width 
    if X[-1] < last: 
        X.append(last)
    Y = [y for y in range(0, matrix.shape[1] - height +1, 60)]
    last = matrix.shape[1] - height 
    if Y[-1] < last: 
        Y.append(last)
    

    for i in X: 
        for j in Y: 
            #print(i, j)
            print("matrix : ", i, " of ", matrix.shape[0], " ",  j, " of ", matrix.shape[1], "            ", end="\r")
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            window = window_and_box[0] 
            window0 = pp.transform(window, methods[0], resolution) 
            window1 = pp.transform(window, methods[1], resolution)
            window0 = np.reshape(window0, (1, width, height, 1))
            window1 = np.reshape(window1, (1, width, height, 1))
            pred = model.predict([window0, window1])
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    p = max(label_matrix[i + x][j + y], pred[x][y]) 
                    #if p >= 0.99: 
                    label_matrix[i + x][j + y] = p 
    #predictions.append(label_matrix) 
         
    #label_matrix = np.rot90(np.zeros(predictions[0].shape), k=1) 

    #for i in range(0, 4): 
    #    predictions[i] = np.rot90(predictions[i], k=3-i)  
    #    label_matrix = label_matrix + predictions[i] 
        
    #label_matrix /= 4
    #label_matrix = np.rot90(label_matrix, k=1)
    clear_session()
    print()
    return label_matrix




def predict_multi_input_frames_rotated(npyname, weights, methods, width=64, height=64, smallsize=0, model=None, resolution=2):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session
    print(npyname)
    if model == None:
        model = create_multiscale_segmentation_model(width, height)
    model.load_weights(weights)
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    """
    matrix = smallsize_matrix_general(m, 2)[0] 
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)
    """
    if smallsize == 1: 
        matrix = smallsize_matrix_general(m, 2)[0] 
    else: 
        matrix = m
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)

    data_list = [[matrix0, matrix1, matrix2, matrix3]]
    #data_list = pp.transform_dataset_list(data_list, method)
    matrix0 = data_list[0][0]
    matrix1 = data_list[0][1] 
    matrix2 = data_list[0][2] 
    matrix3 = data_list[0][3]
    #plt.imshow(matrix0) 
    #plt.show()
    predictions = []
    m_index = 0
    
    for matrix in [matrix0, matrix1, matrix2, matrix3]: 
        m_index += 1
        label_matrix = np.zeros(matrix.shape)
        X = [x for x in range(0, matrix.shape[0] - width + 1, 60)]
        last = matrix.shape[0] - width 
        if X[-1] < last: 
            X.append(last)
        Y = [y for y in range(0, matrix.shape[1] - height +1, 60)]
        last = matrix.shape[1] - height 
        if Y[-1] < last: 
            Y.append(last)
        #print(X[-1], Y[-1])
        #print(X[-1] + width, Y[-1] + height)
        #print(matrix.shape[0], matrix.shape[1])

        for i in X: 
            for j in Y: 
                #print(i, j)
                print("matrix ", m_index, ": ", i, " of ", matrix.shape[0], " ",  j, " of ", matrix.shape[1], "            ", end="\r")
                window_and_box = get_window_by_position(width, height, matrix, i, j) 
                window = window_and_box[0] 
                window0 = pp.transform(window, methods[0], resolution) 
                window1 = pp.transform(window, methods[1], resolution)
                window0 = np.reshape(window0, (1, width, height, 1))
                window1 = np.reshape(window1, (1, width, height, 1))
                pred = model.predict([window0, window1])
                pred = np.reshape(pred, (width, height))
                for x in range(pred.shape[0]): 
                    for y in range(pred.shape[1]): 
                        p = max(label_matrix[i + x][j + y], pred[x][y]) 
                        #if p >= 0.99: 
                        label_matrix[i + x][j + y] = p 
        predictions.append(label_matrix) 
         
    label_matrix = np.rot90(np.zeros(predictions[0].shape), k=1) 
    for i in range(0, 4): 
        predictions[i] = np.rot90(predictions[i], k=3-i)  
        label_matrix = label_matrix + predictions[i] 
        
    label_matrix /= 4
    label_matrix = np.rot90(label_matrix, k=1)
    clear_session()
    print()
    return label_matrix

 


def predict_DTM_frames_rotated(dtm, weights, method, width=64, height=64, smallsize=0):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session
    

    #print(npyname)
    model = create_multiscale_segmentation_model(width, height)
    model.load_weights(weights)
    m = dtm.matrix
    #m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    """
    matrix = smallsize_matrix_general(m, 2)[0] 
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)
    """
    if smallsize == 1: 
        matrix = smallsize_matrix_general(m, 2)[0] 
    else: 
        matrix = m
    matrix1, matrix2, matrix3, matrix0 = get_four_rotation_matrices(matrix)

    data_list = [[matrix0, matrix1, matrix2, matrix3]]
    data_list = pp.transform_dataset_list(data_list, method)
    matrix0 = data_list[0][0]
    matrix1 = data_list[0][1] 
    matrix2 = data_list[0][2] 
    matrix3 = data_list[0][3]
    #plt.imshow(matrix0) 
    #plt.show()
    predictions = []
    m_index = 0
    
    for matrix in [matrix0, matrix1, matrix2, matrix3]: 
        m_index += 1
        label_matrix = np.zeros(matrix.shape)
        X = [x for x in range(0, matrix.shape[0] - width + 1, 60)]
        last = matrix.shape[0] - width 
        if X[-1] < last: 
            X.append(last)
        Y = [y for y in range(0, matrix.shape[1] - height +1, 60)]
        last = matrix.shape[1] - height 
        if Y[-1] < last: 
            Y.append(last)
        #print(X[-1], Y[-1])
        #print(X[-1] + width, Y[-1] + height)
        #print(matrix.shape[0], matrix.shape[1])

        for i in X: 
            for j in Y: 
                #print(i, j)
                print("matrix ", m_index, ": ", i, " of ", matrix.shape[0], " ",  j, " of ", matrix.shape[1], "            ", end="\r")
                window_and_box = get_window_by_position(width, height, matrix, i, j) 
                window = window_and_box[0] 
                window = np.reshape(window, (1, width, height, 1))
                pred = model.predict(window)
                pred = np.reshape(pred, (width, height))
                for x in range(pred.shape[0]): 
                    for y in range(pred.shape[1]): 
                        p = max(label_matrix[i + x][j + y], pred[x][y]) 
                        #if p >= 0.99: 
                        label_matrix[i + x][j + y] = p 
        predictions.append(label_matrix) 
         
    label_matrix = np.rot90(np.zeros(predictions[0].shape), k=1) 
    for i in range(0, 4): 
        predictions[i] = np.rot90(predictions[i], k=3-i)  
        label_matrix = label_matrix + predictions[i] 
        
    label_matrix /= 4
    label_matrix = np.rot90(label_matrix, k=1)
    clear_session()
    print()
    dtm.predicted_matrix = label_matrix
    #return label_matrix




def predict_frames(npyname, weights, method, width=64, height=64):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session

    print(npyname)
    #model = create_multiscale_segmentation_model(width, height)
    model = create_rotation_segmentation_model(width, height)
    model.load_weights(weights)
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    
    matrix = smallsize_matrix_general(m, 2)[0] 
    matrix = pp.transform(matrix, method)
    print(matrix.shape)
    matrix = pp.standard_scaling([matrix])[0]
    print(matrix.shape)
    label_matrix = np.zeros(matrix.shape)
    for i in range(0, matrix.shape[0] - width + 1, 12): 
        for j in range(0, matrix.shape[1] - height + 1, 12): 
            #print(i, j)
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            window = window_and_box[0] 
            window = np.reshape(window, (1, width, height, 1))
            #window -= np.min(window)
            pred = model.predict(window)
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    label_matrix[i + x][j + y] = max(label_matrix[i + x][j + y], pred[x][y])  
                    #label_matrix[i + x][j + y] = (label_matrix[i + x][j + y] + pred[x][y]) / 2            

    plt.subplot(2,1,1) 
    plt.imshow(matrix, cmap="Greys") 
    plt.subplot(2,1,2) 
    plt.imshow(label_matrix, cmap="hot") 
    plt.show()
    clear_session()
    #np.save("images/prediction_frame_vieritz_laplace.npy", label_matrix)
    return label_matrix




def predict_multiscale_frames(npyname, weights, method, width=64, height=64, model=None):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session

    print(npyname)
    if model == None:
        model = create_multiscale_segmentation_model(width, height)
    #model = create_rotation_segmentation_model(width, height)
    model.load_weights(weights)
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    
    matrix = smallsize_matrix_general(m, 2)[0] 
    matrix = pp.transform(matrix, method)
    print(matrix.shape)
    matrix = pp.standard_scaling([matrix])[0]
    print(matrix.shape)
    label_matrix = np.zeros(matrix.shape)
    for i in range(0, matrix.shape[0] - width + 1, 12): 
        for j in range(0, matrix.shape[1] - height + 1, 12): 
            #print(i, j)
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            window = window_and_box[0] 
            window = np.reshape(window, (1, width, height, 1))
            #window -= np.min(window)
            pred = model.predict(window)
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    label_matrix[i + x][j + y] = max(label_matrix[i + x][j + y], pred[x][y])  
                    #label_matrix[i + x][j + y] = (label_matrix[i + x][j + y] + pred[x][y]) / 2            

    plt.subplot(2,1,1) 
    plt.imshow(matrix, cmap="Greys") 
    plt.subplot(2,1,2) 
    plt.imshow(label_matrix, cmap="hot") 
    plt.show()
    clear_session()
    #np.save("images/prediction_frame_vieritz_laplace.npy", label_matrix)
    return label_matrix


def predict_multifilter_frames(npyname, width=64, height=64):
    from utils import get_four_rotation_matrices
    from tensorflow.keras.backend import clear_session
    weights = "checkpoints/multifilter_segmentation_model_2021-5-6:0-40-21epoch_00060.h5"    
    
    model = create_multifilter_segmentation_model4(width, height)
    
    model.load_weights(weights)
    m = np.load(npyname)   
    if len(m.shape) > 2: 
        m = m[0]  
    
    matrix = smallsize_matrix_general(m, 2)[0] 
    #matrix = pp.transform(matrix, method)
    #print(matrix.shape)
    #matrix = pp.standard_scaling([matrix])[0]
    #print(matrix.shape)
    label_matrix = np.zeros(matrix.shape)
    for i in range(0, matrix.shape[0] - width + 1, 12): 
        for j in range(0, matrix.shape[1] - height + 1, 12): 
            #print(i, j)
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            window = window_and_box[0] 
            window = np.reshape(window, (1, width, height, 1))
            #window -= np.min(window)
            pred = model.predict(window)
            pred = np.reshape(pred, (width, height))
            for x in range(pred.shape[0]): 
                for y in range(pred.shape[1]): 
                    label_matrix[i + x][j + y] = max(label_matrix[i + x][j + y], pred[x][y])  
                    #label_matrix[i + x][j + y] = (label_matrix[i + x][j + y] + pred[x][y]) / 2            

    plt.subplot(2,1,1) 
    plt.imshow(matrix, cmap="Greys") 
    plt.subplot(2,1,2) 
    plt.imshow(label_matrix, cmap="hot") 
    plt.show()
    clear_session()
    #np.save("images/prediction_frame_vieritz_laplace.npy", label_matrix)
    return label_matrix





def combine_prediction_methods(): 
    checkpoint1 = "checkpoints/multiscale_segmentation_model_shape_index_2021-3-15:13-26-32epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_pseudo_slope_2021-3-14:15-52-11epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_curvedness_2021-3-14:11-56-0epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_mean_curvature_2021-3-13:18-13-49epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_rotor_2021-3-13:14-42-3epoch_00100.h5" # bad results
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-12:4-9-38epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_unsphericity_curvature_2021-3-12:7-38-48epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_gaussian_curvature_2021-3-13:21-45-12epoch_00100.h5"
    #checkpoint2 = "checkpoints/multiscale_segmentation_model_minimal_curvature_2021-3-14:1-17-18epoch_00100.h5"
    checkpoint2 = "checkpoints/multiscale_segmentation_model_maximal_curvature_2021-3-14:4-49-47epoch_00100.h5"
    npyname = "images/vieritz.npy"     
    smallsize = 1
    matrix = np.load(npyname)  
    if len(matrix.shape) > 2: 
        matrix = matrix[0]  
        smallsize = 0
    method1 = "shape_index"
    #method2 = "pseudo_slope"
    #method2 = "curvedness"
    #method2 = "mean_curvature"
    #method2 = "rotor"
    #method2 = "laplacian"
    #method2 = "unsphericity_curvature"
    #method2 = "gaussian_curvature"
    #method2 = "minimal_curvature"
    method2 = "maximal_curvature"
    m1 = pp.transform(np.copy(matrix), method1)
    m2 = pp.transform(np.copy(matrix), method2)
    label_matrix1 = predict_frames_by_saved_weights_rotated(npyname, checkpoint1, method1)
    label_matrix2 = predict_frames_by_saved_weights_rotated(npyname, checkpoint2, method2)
    label_matrix = (label_matrix1 + label_matrix2) / 2.
    plt.subplot(1,3,1) 
    if smallsize == 1: 
        m1 = smallsize_matrix_general(m1, 2)[0]
    plt.imshow(m1, cmap="Greys") 
    plt.subplot(1,3,2) 
    if smallsize == 1: 
        m2 = smallsize_matrix_general(m2, 2)[0]
    m2 = np.clip(m2, -0.5, 0.5)
    plt.imshow(m2, cmap="Greys") 
    plt.subplot(1,3,3) 
    plt.imshow(label_matrix, cmap="hot")        
    plt.show()





    



def dtm_prediction(): 
    from dtm import DTM, Coordinates
    
    path = "images"
    filename = "vieritz"
    file_format = "npy"
    xml_folder = "images/vieritz.xml"
    coordinates = Coordinates(5821381, 315800, 5822065, 316512)
    
    dtm = DTM(path, filename, file_format, 1, voc=False, coordinates=coordinates)
    dtm.set_matrix(1) 
    print(dtm.coordinates.shape)
    print()
    checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-18:19-37-44epoch_00005.h5"
    method = "laplacian"
    smallsize = 1
    predict_DTM_frames_rotated(dtm, checkpoint, method, smallsize=smallsize)

    plt.subplot(1,2,1) 
    if smallsize == 1:
        mat = smallsize_matrix_general(dtm.matrix, 2)[0]
    else: 
        mat = dtm.matrix
    plt.imshow(np.clip(pp.transform(mat, "laplacian"), -1, 1), cmap="Greys") 
    plt.subplot(1,2,2) 
    plt.imshow(dtm.predicted_matrix, cmap="hot")   # (mat / 2) +     
    plt.show()





def show_fixed_multiscale_results(): 
    #checkpoint = "checkpoints/multiscale_segmentation_model_shape_index_2021-3-31:10-16-35epoch_00100.h5"
    #checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-28:17-48-46epoch_00100.h5"
    checkpoint = "checkpoints/multiscale_segmentation_model_reflectance_2021-3-28:9-37-3epoch_00100.h5"
    npyname = "images/vieritz.npy"
    
    matrix = np.load(npyname)    
    if len(matrix.shape) > 2: 
        matrix = matrix[0]     
    #method = "pseudo_slope"
    #method = "shape_index"
    #method = "rotor"
    #method = "curvedness"
    #method = "laplacian"
    #method = "slrm_cv2_average"
    #method = "slope"
    method = "reflectance"
    #method = "cv2_sobel2D"
    #method = "unsphericity_curvature"
    #method = "sky_view_factor"
    #method = "local_dominance"
    #method = "sky_illumination"
    matrix = pp.transform(matrix, method)
    #vieritz = pp.standard_scaling([vieritz])[0]
    plt.imshow(matrix, cmap="Greys")
    plt.show()
    #label_matrix = predict_frames(npyname, checkpoint, "laplacian")
    model = create_multiscale_segmentation_model3(64, 64)
    #label_matrix = predict_multiscale_frames(npyname, checkpoint, method, model=model)
    if npyname == "images/vieritz.npy": 
        smallsize = 1 
    else: 
        smallsize = 0
    label_matrix = predict_frames_by_saved_weights_rotated(npyname, checkpoint, method, smallsize=smallsize, model=model)
    print(matrix.shape)
    plt.subplot(1,2,1) 
    if npyname == "images/vieritz.npy":
        mat = smallsize_matrix_general(matrix, 2)[0]
    else: 
        mat = matrix
    plt.imshow(np.clip(mat, -1, 1), cmap="Greys") 
    plt.subplot(1,2,2) 
    plt.imshow(label_matrix, cmap="hot")   # (mat / 2) +     
    plt.show()



def predict_multi_input(): 
    #checkpoint = ""
    #checkpoint = "checkpoints/multi_input_segmentation_model6_unsphericity_curvature_shape_index_2021-4-3:4-16-22_epoch_00060.h5" # schlecht
    #checkpoint = "checkpoints/multi_input_segmentation_model6_unsphericity_curvature_shape_index_2021-4-5:11-10-47_epoch_00060.h5" # gut
    #checkpoint = "checkpoints/multi_input_segmentation_model6_reflectance_shape_index_2021-4-3:11-54-55_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_unsphericity_curvature_aspect_2021-4-2:23-14-9_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_unsphericity_curvature_sky_view_factor_2021-4-3:1-55-44_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_cv2_sobel2D_reflectance_2021-4-2:20-54-29_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_cv2_sobel2D_unsphericity_curvature_2021-4-2:18-18-22_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_slrm_cv2_average_shape_index_2021-4-2:15-22-57_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_slrm_cv2_average_sky_view_factor_2021-4-2:13-3-28_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_laplacian_sky_view_factor_2021-4-1:6-19-4_epoch_00060.h5"
    checkpoint = "checkpoints/multi_input_segmentation_model6_laplacian_reflectance_2021-4-1:3-51-21_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_slope_shape_index_2021-4-2:5-59-17_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_slope_reflectance_2021-4-2:3-39-23_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_slope_unsphericity_curvature_2021-4-1:20-45-22_epoch_00060.h5"
    #methods = ["unsphericity_curvature", "shape_index"]
    #methods = ["reflectance", "shape_index"] # sehr gut
    #methods = ["unsphericity_curvature", "aspect"] # sehr gut (rotated), gut  
    #methods = ["unsphericity_curvature", "sky_view_factor"] # gut
    #methods = ["cv2_sobel2D", "reflectance"] # schlecht
    #methods = ["cv2_sobel2D", "unsphericity_curvature"] # schlecht
    #methods = ["slrm_cv2_average", "shape_index"] # error
    #methods = ["slrm_cv2_average", "sky_view_factor"] # error
    #methods = ["laplacian", "sky_view_factor"] # gut, viele false positives
    methods = ["laplacian", "reflectance"] # sehr gut, gut 
    #methods = ["slope", "shape_index"] # gut, relativ viele false positives (rotated), sehr gut
    #methods = ["slope", "reflectance"] # sehr gut (rotated), gut, viele false positives
    #methods = ["slope", "unsphericity_curvature"] # gut (rotated), sehr gut
    #methods = []

    # new filters:
    #checkpoint = "checkpoints/multi_input_segmentation_model6_rotor_pseudo_slope_2021-5-25:7-53-7_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_rotor_minimal_curvature_2021-5-25:5-34-50_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_rotor_reflectance_2021-5-24:22-32-38_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_rotor_laplacian_2021-5-24:20-13-47_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_rotor_slope_2021-5-24:17-55-35_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_pseudo_slope_laplacian_2021-5-24:4-2-25_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_pseudo_slope_mean_curvature_2021-5-24:10-58-35_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_pseudo_slope_shape_index_2021-5-24:8-40-1_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_minimal_curvature_mean_curvature_2021-5-24:1-31-39_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_minimal_curvature_mean_curvature_2021-5-24:1-31-39_epoch_00040.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_minimal_curvature_shape_index_2021-5-23:23-12-49_epoch_00060.h5"
    #checkpoints = "checkpoints/multi_input_segmentation_model6_minimal_curvature_laplacian_2021-5-23:18-33-41_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_minimal_curvature_reflectance_2021-5-23:20-53-14_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_maximal_curvature_reflectance_2021-5-23:12-35-58_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_mean_curvature_reflectance_2021-5-23:3-18-38_epoch_00060.h5"
    #checkpoint = "checkpoints/multi_input_segmentation_model6_mean_curvature_slope_2021-5-22:22-40-37_epoch_00060.h5"
    #methods = ["rotor", "pseudo_slope"] # ganz schlecht, erkennt gar nichts.
    #methods = ["rotor", "minimal_curvature"]
    #methods = ["rotor", "reflectance"] #viele false positives 
    #methods = ["rotor", "laplacian"] # sehr gut
    #methods = ["rotor", "slope"] # viele false positives
    #methods = ["pseudo_slope", "laplacian"] # sehr gut, zu den Favoriten.
    #methods = ["pseudo_slope", "mean_curvature"] #gut, viele false positives
    #methods = ["pseudo_slope", "shape_index"] # sehr gut
    #methods = ["minimal_curvature", "mean_curvature"] # sehr gut, zu Favoriten.
    #methods = ["minimal_curvature", "shape_index"]
    #methods = ["minimal_curvature", "laplacian"] # unbrauchbar
    #methods = ["minimal_curvature", "reflectance"] #gut, recht viele false positives
    #methods = ["maximal_curvature", "reflectance"] # sehr gut, zu Favoriten
    #methods = ["mean_curvature", "reflectance"] # sehr schlecht
    #methods = ["mean_curvature", "slope"]
    #methods = []


    #predict_multi_input_frames_rotated(npyname, weights, methods, width=64, height=64, smallsize=0, model=None, resolution=2)
    npyname = "images/vieritz.npy"
    
    matrix = np.load(npyname)    
    if len(matrix.shape) > 2: 
        matrix = matrix[0]     
    
    matrix = pp.transform(matrix, "pseudo_slope" ) #  "minimal_curvature"
    #plt.imshow(matrix, cmap="Greys")
    #plt.show()

    model = create_multi_input_model6(64, 64) 
    
    if npyname == "images/vieritz.npy": 
        smallsize = 1 
    else: 
        smallsize = 0
    #label_matrix = predict_frames_by_saved_weights_rotated(npyname, checkpoint, method, smallsize=smallsize, model=model)
    label_matrix = predict_multi_input_frames(npyname, checkpoint, methods, width=64, height=64, smallsize=smallsize, model=model, resolution=2)
    #label_matrix = predict_multi_input_frames_rotated(npyname, checkpoint, methods, width=64, height=64, smallsize=smallsize, model=model, resolution=2)
    print(matrix.shape)
    plt.subplot(1,2,1) 
    if npyname == "images/vieritz.npy":
        mat = smallsize_matrix_general(matrix, 2)[0]
    else: 
        mat = matrix
    plt.imshow(np.clip(mat, -1, 1), cmap="Greys") 
    plt.subplot(1,2,2) 
    plt.imshow(label_matrix, cmap="hot")   # (mat / 2) +     
    plt.show()



def predict_multifilter_detection(matrix, steps, verbose=True, model=None, threshold=0.5, tolerance=1, mode="D1", return_circle_list=False): 
    from preprocessing import transform
    from utils import add_bounding_circles_from_pcr, bbox_management2, bbox_management, remove_outliers_from_matrix
    from augment_by_smallsizing import smallsize_matrix_general
    #npyname = "images/vieritz.npy"
    #matrix = remove_outliers_from_matrix(matrix, 2, mean_type="median")
    #matrix -= np.min(matrix)     
    #matrix = matrix.clip(-5, 5)
    #matrix = matrix[168:,:] 
    #matrix = matrix[:,:322]
    #matrix = matrix[matrix.shape[0]-300:,:322]
    #width = matrix.shape[0] 
    #height = matrix.shape[1]
    width = 64 
    height = 64
    if model == None:
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-7:2-51-22epoch_00100.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-7:22-51-44epoch_00100.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-9:0-37-39epoch_00050.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-11:13-7-45epoch_00100.h5" # MAE
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-11:22-53-27epoch_00080.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-13:10-8-12epoch_00100.h5"# bigger segmentation dataset
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-13:20-1-41epoch_00100_no_transfer.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-14:10-53-15epoch_00100.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-15:1-51-4epoch_00020.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-17:10-3-36epoch_00060.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-16:2-5-23epoch_00080.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_2021-5-19:1-33-32epoch_00100.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_15_2021-5-28:0-16-9epoch_00100.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_15_2021-5-28:23-0-30epoch_00020.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00040.h5" 
        #checkpoint = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00040.h5" 
        #checkpoint = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00100.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_18_2021-6-1:12-44-46epoch_00090.h5"
        #checkpoint = "checkpoints/multifilter_detection_model_18_2021-6-6:11-26-13epoch_00100.h5"   
        if mode == "D1":
            checkpoint = "checkpoints/multifilter_detection_model_18_2021-6-8:9-49-38epoch_00020.h5" # open, no lower saxony    
        if mode == "D2":
            #checkpoint = "checkpoints/multifilter_detection_model_22_2021-6-9:10-26-2epoch_00090.h5" # half open
            #checkpoint = "checkpoints/multifilter_detection_model_22_2021-6-9:23-47-49epoch_00010.h5" # half open, rotated with wrong segmentation checkpoint
            checkpoint = "checkpoints/multifilter_detection_model_22_2021-6-10:12-17-39epoch_00090.h5" # half open, rotated
            #if mode == "tversky":
            #checkpoint = "checkpoints/multifilter_detection_model_23_2021-7-6:22-52-57epoch_00010.h5"
            #print("Checkpoint: ", checkpoint)
            
        
        #model.load_weights()
        if mode == "1":
            model = create_multifilter_detection_model3()  # model 1 in MA
        #model = create_multifilter_detection_model4()
        #model = create_multifilter_detection_model5() # MAE
        #model = create_multifilter_detection_model6()
        #model = create_multifilter_detection_model7() # bigger segmentation dataset
        #model = create_multifilter_detection_model8() 
        #model = create_multifilter_detection_model10()
        #model = create_multifilter_detection_model11()
        #model = create_multifilter_detection_model12()
        #model = create_multifilter_detection_model13()
        if mode == "2":
            model = create_multifilter_detection_model14(width=64, height=64)  # model 2 in MA
        #model = create_multifilter_detection_model15(width=64, height=64)
        #model = create_multifilter_detection_model16(width=64, height=64) 
        if mode == "3":
            model = create_multifilter_detection_model17(width=64, height=64) # model 3 in MA  #Score:  [0.19012552926831303, 0.12903444807125472, 0.06109108128840009]
        #model = create_multifilter_detection_model18(width=64, height=64)
        #model = create_multifilter_detection_model20(width=64, height=64)
        if mode == "D1":
            model = create_multifilter_detection_model21(width=64, height=64) # open
        if mode == "D2":            
            model = create_multifilter_detection_model22(width=64, height=64) # half open
            #model = create_multifilter_detection_model23_weighed(width=64, height=64)

        
        model.load_weights(checkpoint)

    if steps != 0: 
        matrix = smallsize_matrix_general(matrix, steps)[0] 
    label_matrix_class = np.zeros(matrix.shape)
    label_matrix_reg   = np.zeros(matrix.shape)

    #stride = 12 # standard for model 20
    stride = 30

    X = [x for x in range(0, matrix.shape[0] - width + 1, stride)]
    last = matrix.shape[0] - width 
    if X[-1] < last: 
        X.append(last)
    Y = [y for y in range(0, matrix.shape[1] - height +1, stride)]
    last = matrix.shape[1] - height 
    if Y[-1] < last: 
        Y.append(last)


    #for i in range(0, matrix.shape[0] - width + 1, 30): 
    #    for j in range(0, matrix.shape[1] - height + 1, 30): 
    for i in X: 
        for j in Y: 
            #print(i, j, " bis hierher ok 1.")       
            window_and_box = get_window_by_position(width, height, matrix, i, j) 
            #print(i, j, " bis hierher ok 2.")
            window = window_and_box[0] 
            window = np.reshape(window, (1, width, height, 1))            
            pred = model.predict(window, batch_size=1)
            #print(i, j, " bis hierher ok 3.")
            #print(pred.shape)

            for k in range(len(pred)):
                #print("Shape: ", pred[k].shape)
                pred[k] = np.reshape(pred[k], (width, height))
            #print(i, j, " bis hierher ok 4.")
            for x in range(pred[0].shape[0]): 
                for y in range(pred[0].shape[1]): 
                    lm_c = label_matrix_class[i + x][j + y]
                    p = pred[0][x][y]
                    r = pred[1][x][y]
                    if p > lm_c: 
                        label_matrix_class[i + x][j + y] = p 
                        label_matrix_reg[i + x][j + y] = r
                    #label_matrix[i + x][j + y] = max(label_matrix[i + x][j + y], pred[x][y]) 
                    #print(i, j, x, y, "            ")   #, end="\r"  
            #print(i, j, " bis hierher ok 5.", matrix.shape)
    #print("bis hierher ok 6.")
    #circle_matrix = bbox_management2(matrix, label_matrix_class, label_matrix_reg, 1, threshold=0.9, tolerance=0.4)
    
    #circle_matrix = bbox_management(matrix, label_matrix_class, label_matrix_reg, 1, threshold=0.9, tolerance=0.9)
    
    m = transform(matrix, "laplacian")#"pseudo_slope"
    m = np.clip(m, -2, 2)
    matrix = np.clip(matrix, 70, 100)
    if return_circle_list == True:
        circle_matrix, circle_list = bbox_management(matrix, label_matrix_class, label_matrix_reg, 1, threshold=threshold, tolerance=tolerance, circle_value=np.max(m), return_circle_list=True)
        print(circle_list)
    else:
        circle_matrix = bbox_management(matrix, label_matrix_class, label_matrix_reg, 1, threshold=threshold, tolerance=tolerance, circle_value=np.max(m), return_circle_list=False)
    print(np.min(m), np.max(m))
    print(np.min(matrix), np.max(matrix))
    
    if verbose == True:    
        #print("bis hierher ok 7.")
        #label_m = add_bounding_circles_from_pcr(m.copy(), label_matrix_class, label_matrix_reg, 10, threshold=0.02)
        label_m = m + circle_matrix / np.max(m)
        #print("bis hierher ok 8.")
        plt.subplot(1,2,1) 
        plt.imshow(m, cmap="Greys") 
        plt.subplot(1,2,2) 
        plt.imshow(label_m, cmap="Greys") # label_matrix_class "hot"
        #print("bis hierher ok 9.")
        plt.show()
    if return_circle_list == True:
        return circle_matrix, circle_list
    else: 
        return circle_matrix
    
                    
    


def predict_multifilter_detection_flexible_size(npyname, steps, first_element_only=False): 
    from preprocessing import transform
    from utils import add_bounding_circles_from_pcr, bbox_management
    from augment_by_smallsizing import smallsize_matrix_general
    #npyname = "images/vieritz.npy"
    matrix = np.load(npyname)  
    if first_element_only: 
        matrix = matrix[0]
    #matrix -= np.min(matrix)     
    #matrix = matrix.clip(-5, 5)
    #matrix = matrix[168:,:] 
    #matrix = matrix[:,:322]
    #matrix = matrix[matrix.shape[0]-300:,:322]
    #width = matrix.shape[0] 
    #height = matrix.shape[1]
    width = matrix.shape[0] 
    height = matrix.shape[1]
    #checkpoint = "checkpoints/multifilter_detection_model_2021-5-7:2-51-22epoch_00100.h5"
    checkpoint = "checkpoints/multifilter_detection_model_2021-5-7:22-51-44epoch_00100.h5" # standard
    #checkpoint = "checkpoints/multifilter_detection_model_2021-5-9:0-37-39epoch_00200.h5"
    model = create_multifilter_detection_model3() # standard
    #model = create_multifilter_detection_model4()
    model.load_weights(checkpoint)
    model.summary()
    if steps != 0: 
        matrix = smallsize_matrix_general(matrix, steps)[0] 
    label_matrix_class = np.zeros(matrix.shape)
    label_matrix_reg   = np.zeros(matrix.shape)
    #for i in range(0, matrix.shape[0] - width + 1, 12): 
    #    for j in range(0, matrix.shape[1] - height + 1, 12): 
            #print(i, j, " bis hierher ok 1.")       
            #window_and_box = get_window_by_position(width, height, matrix, i, j) 
            #print(i, j, " bis hierher ok 2.")
            #window = window_and_box[0] 
    matrix = np.reshape(matrix, (1, matrix.shape[0], matrix.shape[1], 1))            
    pred = model.predict(matrix, batch_size=1)
            #print(i, j, " bis hierher ok 3.")
            #print(pred.shape)

    for k in range(len(pred)):
        pred[k] = np.reshape(pred[k], (width, height))
            #print(i, j, " bis hierher ok 4.")
    for x in range(pred[0].shape[0]): 
        for y in range(pred[0].shape[1]): 
            lm_c = label_matrix_class[i + x][j + y]
            p = pred[0][x][y]
            r = pred[1][x][y]
            if p > lm_c: 
                label_matrix_class[i + x][j + y] = p 
                label_matrix_reg[i + x][j + y] = r
    matrix = matrix.reshape((matrix.shape[1], matrix.shape[2]))                
    circle_matrix = bbox_management(matrix, label_matrix_class, label_matrix_reg, 1, threshold=0.9, tolerance=0.99)
    m = transform(matrix, "laplacian")
    print("bis hierher ok 7.")
    #label_m = add_bounding_circles_from_pcr(m.copy(), label_matrix_class, label_matrix_reg, 10, threshold=0.02)
    label_m = m + circle_matrix
    print("bis hierher ok 8.")
    plt.subplot(1,2,1) 
    plt.imshow(np.clip(m, -20, 20), cmap="Greys") 
    plt.subplot(1,2,2) 
    plt.imshow(label_m, cmap="Greys") # label_matrix_class "hot"
    print("bis hierher ok 9.")
    plt.show()
        

        
            
def predict_numpy_list( model_name, 
                        checkpoint_name, 
                        file_list, 
                        mode, 
                        smallsize, 
                        rotate=False, 
                        first_element_only=False, 
                        threshold=0.8, 
                        tolerance=1., 
                        d_mode="D1", 
                        verbose=False,
                        save_dir="None", 
                        save_image=False): 
    # prediction with a list of numpy-arrays as inputs
    # file_list: list with the paths and names of the numpy_arrays to process
    # mode: segmentation or detection 
    import glob 
    from utils import split_matrix, merge_matrix 
    from augment_by_smallsizing import smallsize_matrix_general
    from utils import save_rgb_image_from_numpy_matrices    

    if save_image == True: 
        from cv2 import cvtColor, COLOR_GRAY2BGR, imwrite 
        from preprocessing import transform

    # load model
    if mode == "segmentation":
        model = model_name(64, 64)        
    elif mode == "detection": 
        if model_name != "":
            model = model_name(width=64, height=64)  
        else: 
            model = None      
    else: 
        raise Exception("Wrong mode! It has to be eather 'segmentation' or 'detection'!")

    # load weights into model
    if checkpoint_name != "":
        model.load_weights(checkpoint_name) 

    index = 1
    for npyname in file_list: 
        print(index, " of ", len(file_list)) 
        index += 1

        if mode == "segmentation": 
        
            if rotate == True:
                pred_matrix = predict_multifilter_segmentation_rotated(npyname, model, smallsize=smallsize, verbose=verbose)     
            else: 
                pred_matrix = predict_multifilter_segmentation(npyname, model, smallsize=smallsize, verbose=verbose)            
        
        elif mode == "detection": 
            matrix = np.load(npyname)  
            if first_element_only: 
                matrix = matrix[0]
            #print("Shape: ", matrix.shape)
            pred_matrix, rc_list = predict_multifilter_detection(matrix, 
                                                    smallsize,
                                                    verbose=verbose,
                                                    model=model, 
                                                    threshold=threshold, 
                                                    tolerance=tolerance,
                                                    mode=d_mode,                                                      
                                                    return_circle_list=True)
            
        #  matrix, steps, verbose=True, model=None, threshold=0.5, tolerance=1, mode="D1", return_circle_list=False

        if save_dir != "None": 
            name = npyname.split("/")[-1].replace(".npy", "_predicted_{}.npy".format(mode)) 
            f_name = save_dir + name
            print(f_name)
            print()
            np.save(f_name, pred_matrix)
            if save_image == True: 

                matrix = np.load(npyname)  
                matrix = smallsize_matrix_general(matrix, smallsize)[0]              
                matrix1 = transform(matrix, "pseudo_slope")
                matrix2 = transform(matrix, "laplacian")
                

                save_rgb_image_from_numpy_matrices(f_name.replace(".npy", ".png"), [matrix1, matrix2, pred_matrix])


            if mode == "detection": 
                cl_path = npyname.split("/")[-1].replace(".npy", "_predicted_bounding_circles.csv")
                cl_name = save_dir + cl_path
                print(f_name)
                print()
                with open(cl_name, "w") as cl_file:
                    for element in rc_list: 
                        cl_file.write("{},{},{},{}\n".format(element[0], element[1], element[2], element[3]))
                


     


if __name__ == "__main__": 


    args = sys.argv[1:]
    #predict_norwegian_data_multifilter(verbose=False, save_dir="images/norwegian_circles/") 
    #sys.exit()
    #predict_multi_input()
    #sys.exit()
    #saxony_detection() 
    #save_saxony_results_as_rgb()
    #sys.exit()
    #dtm_prediction()
    #show_fixed_multiscale_results() 
    #predict_multi_input()
    d_mode = ""
    if "S1" in args: 
        d_mode = "S1" 
    elif "S2" in args: 
        d_mode = "S2" 
    elif "D1" in args: 
        d_mode = "D1" 
    elif "D2" in args: 
        d_mode = "D2"
    else: 
        for a in args: 
            #print(a)
            if "d_mode" in a:                 
                m = a.split("=")[1] 
                d_mode = m 
                break
    print(d_mode)

    
    npyname = "images/vieritz.npy" 
    
        

    if "segmentation" in args:    
    
        if "model=8" in args:
            model = create_multifilter_segmentation_model8(64, 64) 
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:1-48-1epoch_00100.h5") 
        elif "model=9" in args: 
            model = create_multifilter_segmentation_model9(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:19-28-45epoch_00100.h5")
        elif "model=11" in args:
            model = create_multifilter_segmentation_model11(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-17:1-10-20epoch_00120.h5")        
        elif "model=12beta" in args:
            model = create_multifilter_segmentation_model12_beta(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-22:12-8-46epoch_00100.h5")
        elif "model=12" in args:                
            # Model 2 segmentation part
            model = create_multifilter_segmentation_model12(64, 64)            
            #model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-7-2:19-20-0epoch_00100.h5") # mini
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-18:13-35-35epoch_00060_copy.h5") # uncorrected, used for detection.            #
            #model.load_weights("checkpoints/multifilter_segmentation_model_12_2021-6-3:13-3-33epoch_00100.h5")  # bigger2 corrected 
            #model.load_weights("checkpoints/multifilter_segmentation_model_12_2021-6-3:13-3-33epoch_00060.h5")  # bigger2 corrected  
        elif "model=14" in args:
            model = create_multifilter_segmentation_model14(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-26:19-26-24epoch_00100.h5")
            #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-26:19-26-24epoch_00060.h5")
        elif "model=16" in args:
            model = create_multifilter_segmentation_model16(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:0-55-32epoch_00060.h5")
        elif "model=17" in args:
            model = create_multifilter_segmentation_model17(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:11-56-26epoch_00080.h5")
        elif "model=19" in args:
            model = create_multifilter_segmentation_model19(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:18-24-11epoch_00100_copy.h5")             
        elif "model=20" in args:
            model = create_multifilter_segmentation_model20(64, 64)
            model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-28:10-41-44epoch_00100.h5")            
        elif "model=21" in args:
            model = create_multifilter_segmentation_model21(64, 64)
            if not "open" in args:
                checkpoints = "checkpoints/multifilter_segmentation_model_2021-6-2:11-50-12epoch_00060.h5" # bigger2 corrected
                #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-7:19-6-28epoch_00080.h5"
            #model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-6-3:18-42-22epoch_00100.h5") # nls
            #model.load_weights("checkpoints/DTM1_multifilter_segmentation_model_5_2021-6-5:14-3-48epoch_00040.h5") # DTM1
            else: 

                #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-5:23-46-58epoch_00100.h5" # open
                #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-7:19-6-28epoch_00100.h5"
                #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-8:22-15-5epoch_00080.h5" # halv open
                print("model 21 open")
                checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-23:14-46-59epoch_00080.h5" # with learning rate decay

            model.load_weights(checkpoints)
        elif "model=5" in args:
            model = create_multifilter_segmentation_model5(64, 64) 
            checkpoints = "checkpoints/multifilter_detection/multifilter_segmentation_model_2021-5-7:14-21-19epoch_00100.h5" # uncorr, used for detection
            #checkpoints = "checkpoints/multifilter_segmentation_model_5_2021-6-3:1-24-56epoch_00060.h5"
            #checkpoints = "checkpoints/multifilter_segmentation_model_5_2021-6-4:10-25-3epoch_00040.h5"
            model.load_weights(checkpoints)
        elif "model=21_tversky" in args: 
            model = create_multifilter_segmentation_model21_tversky(64, 64)
            #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-21:15-44-44epoch_00020.h5" # alpha = 0.3
            #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-21:15-44-44epoch_00040.h5"

            #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-21:20-51-10epoch_00040.h5" # alpha = 0.7
            checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-22:2-13-57epoch_00020.h5" # alpha = 0.3, with learning rate decay, Score:  [0.25343769607211547, 0.7279181650614768, 0.9935679580226089]
            model.load_weights(checkpoints)

        elif "model=21_mcc" in args: 
            checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-22:17-32-45epoch_00100.h5" # with learning rate decay, Score:  [0.10053871111363492, 0.7175058440257797, 0.9932683980127001]
            
            model = create_multifilter_segmentation_model21_weighed_tversky(64, 64)
            model.load_weights(checkpoints)

        elif "model=mini" in args: 
            model = create_multifilter_segmentation_model21(64, 64)
            #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-1:18-23-12epoch_01000.h5"
            #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-1:19-34-14epoch_00100.h5"
            checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-3:10-55-15epoch_00100.h5" # f1_2 Score:  [f1_2_loss: 0.22649625829748204, f1_2: 0.7735037407544147, acc: 0.992798951295045, precision: 0.8557195134192437, recall: 0.8740694900609653]
            print("Checkpoints: ", checkpoints)
            model.load_weights(checkpoints)

        elif "model=model21_f_beta" in args: 
            model = create_multifilter_segmentation_model21_f_beta(64, 64)
            if "mini" in args:
                # with mini segmentation dataset                
                #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-3:13-20-29epoch_00100.h5" # beta = 3
                #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-3:13-42-51epoch_00100.h5" # beta = 0.5
                checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-3:14-8-3epoch_00100.h5" # beta = 0.2 
            else: 
                # with S2
                checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-3:15-4-39epoch_00100.h5" # beta = 0.2 
            model.load_weights(checkpoints)
        
        elif "model=model21_fowlkes_mallows" in args: 
            model = create_multifilter_segmentation_model21_fowlkes_mallows(64, 64)
            checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-8:23-44-7epoch_00100.h5"
            model.load_weights(checkpoints)

        elif "model=23" in args: 
            model = create_multifilter_segmentation_model23(64, 64)            
            checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-7-1:20-13-43epoch_00100.h5"
            model.load_weights(checkpoints)
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:1-48-1epoch_00100.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-14:19-28-45epoch_00100.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-17:1-10-20epoch_00120.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-18:13-35-35epoch_00060.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-22:12-8-46epoch_00100.h5")
        #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-22:17-26-14epoch_00100.h5")
        #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-25:15-36-30epoch_00060.h5") # alt
        #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-25:20-46-55epoch_00060.h5") # alt
        #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-26:2-4-43epoch_00060.h5")
        #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-26:11-50-29epoch_00060.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-26:19-26-24epoch_00100.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:0-55-32epoch_00060.h5")        
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:11-56-26epoch_00080.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-27:18-24-11epoch_00100.h5")
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-28:10-41-44epoch_00100.h5")        
        #model.load_weights("checkpoints/multifilter_segmentation_model_2021-6-1:2-37-59epoch_00100.h5") # bigger trainingsset, Score:  [0.5463168377933619, 0.45368316225556427, 0.9881276022375879]
        # model.load_weights("checkpoints/multifilter_segmentation_model_2021-6-2:11-50-12epoch_00060.h5")
        #predict_multifilter_segmentation_rotated(npyname, model, smallsize=0)
        if not "vieritz" in args :  #"sachsen" in args or "dtm2" in args
            sachsen = 1
        else: 
            sachsen = 0
        if not sachsen:   
            if "rotate" in args:
                predict_multifilter_segmentation_rotated(npyname, model, smallsize=2)     
            else: 
                predict_multifilter_segmentation(npyname, model, smallsize=2)
        else: 
            if "rotate" in args:
                predict_multifilter_segmentation_rotated(npyname, model, smallsize=0)
            else: 
                predict_multifilter_segmentation(npyname, model, smallsize=0)


    #sys.exit()
    if "detection" in args: 

        if "vieritz" in npyname:
            mode = "vieritz"  
        elif "norway" in npyname: 
            mode = "norway"    
        else: 
            mode = "sachsen"
        #mode = "sachsen"#  "vieritz"
        if mode != "sachsen":
            first_element_only = False
        else: 
            first_element_only = True
        matrix = np.load(npyname)  
        if first_element_only: 
            matrix = matrix[0]

        threshold = 0.25 
        tolerance = 1.0
        for arg in args: 
            if "tolerance" in arg: 
                arg_splitted = arg.split("=") 
                arg_splitted = arg_splitted[1] 
                tolerance = float(arg_splitted) 
            if "threshold" in arg: 
                arg_splitted = arg.split("=") 
                arg_splitted = arg_splitted[1] 
                threshold = float(arg_splitted) 
        print("Threshold: ", threshold, " Tolerance: ", tolerance)
        # vieritz:
        if mode != "sachsen":
            predict_multifilter_detection(matrix, 2, threshold=threshold, tolerance=tolerance, mode=d_mode, return_circle_list=True)
        # sachsen:
        if mode == "sachsen": # or mode == "norway"
            #matrix = matrix[matrix.shape[0]-300:,:322]
            predict_multifilter_detection(matrix, 0, threshold=threshold, tolerance=tolerance, mode=d_mode)
        #predict_multifilter_detection_flexible_size(npyname, 0, first_element_only=True)
    else: 
        

        #checkpoint = "checkpoints/multiscale_segmentation_model_slope_2021-2-28:0-50-46epoch_00080.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-1:0-8-27epoch_00040.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_sky_view_factor_2021-3-1:3-0-33epoch_00060.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_cv2_sobel2D_2021-2-28:21-23-47epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_sky_view_factor_2021-3-1:3-0-33epoch_00060.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_reflectance_2021-2-28:6-19-54epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_reflectance_2021-3-2:11-49-28epoch_00060.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-2:13-35-8epoch_00060.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-18:12-2-19epoch_00005.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-18:19-37-44epoch_00005.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_reflectance_2021-3-5:20-55-52epoch_00040.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_shape_index_2021-3-14:8-22-59epoch_00100.h5" # best
        #checkpoint = "checkpoints/multiscale_segmentation_model_shape_index_2021-3-15:13-26-32epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_pseudo_slope_2021-3-14:15-52-11epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_rotor_2021-3-13:14-42-3epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_curvedness_2021-3-14:11-56-0epoch_00100.h5"

        #checkpoint = "checkpoints/multiscale_segmentation_model_laplacian_2021-3-28:17-48-46epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_slrm_cv2_average_2021-3-28:3-26-40epoch_00100.h5"
        #checkpoint = "checkpoints/"
        #checkpoint = "checkpoints/multiscale_segmentation_model_slope_2021-3-27:22-19-6epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_reflectance_2021-3-28:9-37-3epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_cv2_sobel2D_2021-3-28:13-42-37epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_unsphericity_curvature_2021-3-28:21-54-47epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_sky_view_factor_2021-3-29:2-9-54epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_local_dominance_2021-3-29:16-35-54epoch_00100.h5"
        #checkpoint = "checkpoints/multiscale_segmentation_model_sky_illumination_2021-3-29:10-22-54epoch_00080.h5"
        #npyname = "images/vieritz.npy"
        
        vieritz = np.load(npyname)
        #vieritz = pp.transform(vieritz, "slope")
        #vieritz = pp.transform(vieritz[0], "reflectance")
        #if len(vieritz.shape) > 2: 
            #vieritz = vieritz[0] 
        #vieritz = pp.transform(vieritz, "laplacian")
        #vieritz = pp.transform(vieritz, "reflectance")
        #method = "pseudo_slope"
        #method = "shape_index"
        #method = "rotor"
        #method = "curvedness"
        #method = "laplacian"
        #method = "slrm_cv2_average"
        #method = "slope"
        #method = "reflectance"
        #method = "cv2_sobel2D"
        #method = "unsphericity_curvature"
        #method = "sky_view_factor"
        #method = "local_dominance"
        method = "sky_illumination"
        vieritz = pp.transform(vieritz, method)
        #vieritz = pp.standard_scaling([vieritz])[0]
        plt.imshow(vieritz, cmap="Greys")
        plt.show()
        #label_matrix = predict_frames(npyname, checkpoint, "laplacian")
        model = create_multiscale_segmentation_model3(64, 64)
        #label_matrix = predict_multiscale_frames(npyname, checkpoint, method, model=model)
        if npyname == "images/vieritz.npy": 
            smallsize = 1 
        else: 
            smallsize = 0
        label_matrix = predict_frames_by_saved_weights_rotated(npyname, checkpoint, method, smallsize=smallsize, model=model)
        print(vieritz.shape)
        plt.subplot(1,2,1) 
        if npyname == "images/vieritz.npy":
            mat = smallsize_matrix_general(vieritz, 2)[0]
        else: 
            mat = vieritz
        plt.imshow(np.clip(mat, -1, 1), cmap="Greys") 
        plt.subplot(1,2,2) 
        plt.imshow(label_matrix, cmap="hot")   # (mat / 2) +     
        plt.show()





