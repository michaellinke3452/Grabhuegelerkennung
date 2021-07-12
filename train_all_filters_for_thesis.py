import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import *
import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from augment_by_smallsizing import smallsize_matrix_general
from utils import get_window_by_position, asarray_trainingsset, reshape_trainingsset, get_timestamp
import predictions as pd
import sys

# terminal command for training of single input model: 
# python train_all_filters_for_thesis.py single 

# terminal command for training of multi input model: 
# python train_all_filters_for_thesis.py multi

def train_single_input(method, scaling=True, pca_components=0):
    # Trains a CNN with a single filtered input matrix. 
    # Training with S1 dataset might lead to memory issues. 

    from file_ops import read_h5py_segmentation_dataset
    #from file_ops import read_segmentation_dataset   
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session    
    
    x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius.h5") # used for training    
    #x_train, y_train = read_segmentation_dataset("S1_train.h5", train=True)
    #x_test, y_test = read_segmentation_dataset("S1_test.h5", test=True)


    # Preprocessing 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) 

    x_train, y_train = pp.rotate_dataset(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)    

    x_train, y_train, x_val, y_val, x_test, y_test = asarray_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)
    #plt.imshow(x_train[0], cmap="Greys") 
    #plt.show()
    pp.transform_dataset_list([x_train, x_val, x_test], method, pca_components=pca_components)  
    if method in ["accumulation_curvature", "cv2_sobel2D", "double_sobel"]: 
        x_train /= 1000 
        x_val /= 1000 
        x_test /= 1000
    
    if scaling == True:
        pp.standard_scaling([x_train, x_val, x_test])  
    #pp.normalize([x_train, x_val, x_test])
    #plt.imshow(x_train[0], cmap="Greys") 
    #plt.show()
    x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)     

    width = x_train[0].shape[0]
    height = x_train[0].shape[1]
    batch_size = 16
    model = create_multiscale_segmentation_model3(width, height)
    #model = create_rotation_segmentation_model(batch_size, width, height)
    model.summary()
    epochs = 100
    

    timestamp = get_timestamp()
    #c_path = "checkpoints/rotation_segmentation_model_" + method + "_" + timestamp
    c_path = "checkpoints/multiscale_segmentation_model_" + method + "_" + timestamp
    checkpoint_path = c_path + "epoch_{epoch:05d}.h5"
    checkpoint_frequency = 20
    callbacks = []
    cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=checkpoint_frequency)
    callbacks.append(cp_callback)
    csv_logger = CSVLogger(c_path + ".log")
    callbacks.append(csv_logger)

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, validation_data=[x_val, y_val])
    score = model.evaluate(x_test, y_test, batch_size=1)

    print("Score: ", score)    
    log_info = "{}, {}, {}, {}\n".format(method, timestamp, epochs, score)    
    clear_session()
    return log_info





def train_single_input_with_shifts(method, width=64, height=64, scaling=True):
    # deprecated. Do not use.
    from file_ops import read_h5py_segmentation_dataset
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session
    
    model = create_multiscale_segmentation_model(width, height)
    model.summary()
    for shift_index in range(1, 100): 
        x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius_" + str(shift_index) + ".h5")
        
        # Preprocessing 
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) 

        x_train, y_train = pp.rotate_dataset(x_train, y_train)
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)    

        x_train, y_train, x_val, y_val, x_test, y_test = asarray_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)
        #plt.imshow(x_train[0], cmap="Greys") 
        #plt.show()
        pp.transform_dataset_list([x_train, x_val, x_test], method)  
        if method in ["accumulation_curvature", "cv2_sobel2D", "double_sobel"]: 
            x_train /= 1000 
            x_val /= 1000 
            x_test /= 1000
        if scaling == True:
            pp.standard_scaling([x_train, x_val, x_test])  
        #pp.normalize([x_train, x_val, x_test])
        #plt.imshow(x_train[0], cmap="Greys") 
        #plt.show()
        x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)     

        #width = x_train[0].shape[0]
        #height = x_train[0].shape[1]
        batch_size = 32        
        epochs = 5
        

        timestamp = get_timestamp()
        #c_path = "checkpoints/rotation_segmentation_model_" + method + "_" + timestamp
        c_path = "checkpoints/multiscale_segmentation_model_" + method + "_" + timestamp
        checkpoint_path = c_path + "epoch_{epoch:05d}.h5"
        checkpoint_frequency = 5
        callbacks = []
        cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=checkpoint_frequency)
        callbacks.append(cp_callback)
        csv_logger = CSVLogger(c_path + ".log")
        callbacks.append(csv_logger)

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, validation_data=[x_val, y_val])
        score = model.evaluate(x_test, y_test, batch_size=1)

        print("Score: ", score)    
    log_info = "{}, {}, {}, {}\n".format(method, timestamp, epochs, score)    
    clear_session()
    return log_info





def train_multi_input(methods, scaling=True):
    # Trains a CNN with two differently filtered input matrices. 
    # training with S1 dataset might lead to memory issues. 
    # methods: pair of filter methods.  
    # Returns a string with log information about: 
    #   method 1, method 2, timestamp, number of epochs, and evaluation score. 

    from file_ops import read_h5py_segmentation_dataset 
    #from file_ops import read_segmentation_dataset   
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session    
    
    x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius.h5")  # used for training
    #x_train, y_train = read_segmentation_dataset("S1_train.h5", train=True)
    #x_test, y_test = read_segmentation_dataset("S1_test.h5", test=True)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) 

    # Preprocessing 

    x_train, y_train = pp.rotate_dataset(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)        
        
    x_train, y_train, x_val, y_val, x_test, y_test = asarray_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)

    x_train2 = np.copy(x_train)
    x_val2 = np.copy(x_val) 
    x_test2 = np.copy(x_test) 

    pp.transform_dataset_list([x_train, x_val, x_test], methods[0]) 
    pp.transform_dataset_list([x_train2, x_val2, x_test2], methods[1])             
    if methods[0] in ["accumulation_curvature", "cv2_sobel2D", "double_sobel"]: 
        x_train /= 1000 
        x_val /= 1000 
        x_test /= 1000
    """    
    if methods[0] in ["aspect"]: 
        x_train /= 2 * np.pi 
        x_val /= 2 * np.pi  
        x_test /= 2 * np.pi 
    """
    if methods[1] in ["accumulation_curvature", "cv2_sobel2D", "double_sobel"]: 
        x_train2 /= 1000 
        x_val2 /= 1000 
        x_test2 /= 1000
    """
    if methods[1] in ["aspect"]: 
        x_train2 /= 2 * np.pi 
        x_val2 /= 2 * np.pi  
        x_test2 /= 2 * np.pi 
    """
    if scaling == True:        
        pp.standard_scaling([x_train, x_val, x_test])
        pp.standard_scaling([x_train2, x_val2, x_test2])

    x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)  
    x_train2 = x_train2.reshape((-1, 64, 64, 1))
    x_val2 = x_val2.reshape((-1, 64, 64, 1)) 
    x_test2 = x_test2.reshape((-1, 64, 64, 1)) 
    
    width = x_train[0].shape[0]
    height = x_train[0].shape[1]

    model = create_multi_input_model6(width, height)           
    model.summary()

    epochs = 60
    batch_size = 16 
    
    timestamp = get_timestamp()
    c_path = "checkpoints/multi_input_segmentation_model6_" + methods[0] + "_" + methods[1] + "_" + timestamp 
    checkpoint_path = c_path + "_epoch_{epoch:05d}.h5"  
    checkpoint_frequency = 20
    callbacks = []         
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=checkpoint_frequency)
    callbacks.append(cp_callback)
    csv_logger = CSVLogger(c_path + ".log")
    callbacks.append(csv_logger)

    model.fit([x_train, x_train2], y_train, epochs=epochs, batch_size=batch_size, shuffle=False, callbacks=callbacks, validation_data=[[x_val, x_val2], y_val])    
    score = model.evaluate([x_test, x_test2], y_test, batch_size=1)

    print("Score: ", score)    
    log_info = "{}, {}, {}, {}, {}\n".format(methods[0], methods[1], timestamp, epochs, score)    
    clear_session()
    return log_info





if __name__ == "__main__": 
    args = sys.argv[1:] 
    #print(args) 
    if "single" in args:
        #print("single") 
        #sys.exit()

        print("Begin Single-Input-Training")

        # methods for one-input-segmentation:        
        
        methods_one_input = [
            "slope", 
            "slrm_cv2_average", 
            "reflectance", 
            "cv2_sobel2D", 
            "laplacian", 
            "unsphericity_curvature", 
            "sky_view_factor",
            "sky_illumination",
            "local_dominance",    

            "pseudo_slope",
            "double_sobel",
            "double_slope",
            "horizontal_curvature",            
            "ring_curvature",
            "rotor",
            "mean_curvature",
            "gaussian_curvature",
            "minimal_curvature",
            "maximal_curvature",
            "shape_index",
            "curvedness",
            "insolation",  
            "vertical_curvature", 
            "difference_curvature", 
            "accumulation_curvature"         
        ]
        

        #methods_one_input = [
        #    "laplacian"
        #]
        #pca_components = 5
        pca_components = 0

        
        # create head for log data file: 
        log_data_single_input = "method, timestamp, epochs, score\n"


        t = get_timestamp()
        log_file = open("checkpoints/log_data_single_input_{}.txt".format(t), "w")
        log_file.write(log_data_single_input)
        log_file.close()
        # train all single-input-methods and save results in log file:


        shift = 0 

        if shift == 0:

            for method in methods_one_input: 
                
                #try:
                print(method)
                log_data_single_input = log_data_single_input + train_single_input(method, scaling=False, pca_components=pca_components)        
                #except: 
                #    log_data_single_input = log_data_single_input + "{}, {}, {}, {}\n".format(method, "error", "error", "error")
                
                log_file = open("checkpoints/log_data_single_input_{}.txt".format(t), "w")
                log_file.write(log_data_single_input)
                log_file.close()

        else: 
            for method in methods_one_input: 
                
                #try:
                print(method)
                log_data_single_input = log_data_single_input + train_single_input_with_shifts(method, scaling=False)        
                #except: 
                #    log_data_single_input = log_data_single_input + "{}, {}, {}, {}\n".format(method, "error", "error", "error")
                
                log_file = open("checkpoints/log_data_single_input_{}.txt".format(t), "w")
                log_file.write(log_data_single_input)
                log_file.close()



    if "multi" in args:          

        print("Begin Multi-Input-Training")        

        # methods for multi-input-segmentation:
        
        methods_multi_input = [
            ["slope", "aspect"], 
            ["slope", "laplacian"], 
            ["slope", "sky_view_factor"], 
            ["slope", "unsphericity_curvature"],
            ["slope", "slrm_cv2_average"],
            ["slope", "cv2_sobel2D"],
            ["slope", "reflectance"],  
            ["slope", "shape_index"],
            ["laplacian", "aspect"], 
            ["laplacian", "slrm_cv2_average"],
            ["laplacian", "unsphericity_curvature"],
            ["laplacian", "cv2_sobel2D"],
            ["laplacian", "reflectance"],
            ["laplacian", "sky_view_factor"],
            ["laplacian", "shape_index"],
            ["slrm_cv2_average", "aspect"], 
            ["slrm_cv2_average", "reflectance"],
            ["slrm_cv2_average", "cv2_sobel2D"],
            ["slrm_cv2_average", "unsphericity_curvature"],
            ["slrm_cv2_average", "sky_view_factor"],
            ["slrm_cv2_average", "shape_index"],
            ["cv2_sobel2D", "aspect"],
            ["cv2_sobel2D", "unsphericity_curvature"],
            ["cv2_sobel2D", "sky_view_factor"],
            ["cv2_sobel2D", "reflectance"],
            ["unsphericity_curvature", "aspect"],
            ["unsphericity_curvature", "reflectance"],
            ["unsphericity_curvature", "sky_view_factor"],
            ["unsphericity_curvature", "shape_index"],
            ["sky_view_factor", "aspect"],
            ["sky_view_factor", "reflectance"],
            ["sky_view_factor", "shape_index"],
            ["reflectance", "aspect"],
            ["reflectance", "shape_index"],
            
            ["mean_curvature", "slope"],
            ["mean_curvature", "laplacian"],
            ["mean_curvature", "reflectance"],
            ["mean_curvature", "shape_index"],
            ["maximal_curvature", "slope"],
            ["maximal_curvature", "laplacian"],
            ["maximal_curvature", "reflectance"],
            ["maximal_curvature", "shape_index"],
            ["maximal_curvature", "mean_curvature"],
            ["minimal_curvature", "slope"],
            ["minimal_curvature", "laplacian"],
            ["minimal_curvature", "reflectance"],
            ["minimal_curvature", "shape_index"],
            ["minimal_curvature", "mean_curvature"], 
            ["minimal_curvature", "maximal_curvature"],
            ["pseudo_slope", "slope"],
            ["pseudo_slope", "laplacian"],
            ["pseudo_slope", "reflectance"],
            ["pseudo_slope", "shape_index"],
            ["pseudo_slope", "mean_curvature"], 
            ["pseudo_slope", "maximal_curvature"],
            ["pseudo_slope", "minimal_curvature"],
            ["rotor", "slope"],
            ["rotor", "laplacian"],
            ["rotor", "reflectance"],
            ["rotor", "shape_index"],
            ["rotor", "mean_curvature"], 
            ["rotor", "maximal_curvature"],
            ["rotor", "minimal_curvature"],
            ["rotor", "pseudo_slope"]
        ]

        # create head for log data file: 
        log_data_multi_input = "method0, method1, timestamp, epochs, score\n"
        t = get_timestamp()
        log_file = open("checkpoints/log_data_multi_input_{}.txt".format(t), "w")
        log_file.write(log_data_multi_input) 
        log_file.close()

        # train all multi-input-methods and save results in log file:
        for methods in methods_multi_input: 
            try:
                print(methods)
                log_data_multi_input = log_data_multi_input + train_multi_input(methods, scaling=False)
            except: 
                log_data_multi_input = log_data_multi_input + "{}, {}, {}, {}, {}\n".format(methods[0], methods[1], "error", "error", "error")
            log_file = open("checkpoints/log_data_multi_input_{}.txt".format(t), "w")
            log_file.write(log_data_multi_input) 
            log_file.close()
        


