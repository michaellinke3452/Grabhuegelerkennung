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



def train_single_input(scaling=False, bigger_test_set=False, rotate=False, model_nr="", lr_decay=False):
    #from file_ops import read_h5py_segmentation_dataset
    from file_ops import read_segmentation_dataset
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session  
    from utils import rotate_segmentation_data  

    print("bigger test set: ", bigger_test_set)
    print("model_nr: ", model_nr) 
    print("lr_decay: ", lr_decay) 

    if bigger_test_set == False:
        # x_train:  12697  x_val:  3175  x_test:  7256
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius.h5")
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_lower_saxony.h5") 
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("OpenSegmentationDatasetSmall.h5")     
        x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("MiniDataset_halv_open.h5") # NOT for publication!!!
    else:
        # x_train:  47744  x_val:  11936  x_test:  29024
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius_bigger2.h5")
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_without_lower_saxony.h5")
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("OpenSegmentationDatasetBig.h5") # standard with errors
        #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("openSegmentationDataset2_big.h5") # for publication
        x_train, y_train = read_segmentation_dataset("S1_train.h5", train=True)
        x_test, y_test = read_segmentation_dataset("S1_test.h5", test=True)
        
    #sys.exit()

    #for i in range(5): 
    #    plt.imshow(x_train[i]) 
    #    plt.show()

    # Preprocessing 
    if bigger_test_set == True and rotate == True:
        x_train, y_train = rotate_segmentation_data(x_train, y_train) # not standard! 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) 
    print("x_train: ", len(x_train), " x_val: ", len(x_val), " x_test: ", len(x_test))
    if bigger_test_set == False:
        x_train, y_train = pp.rotate_dataset(x_train, y_train) #standard for model 1 and 3 
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)    

    x_train, y_train, x_val, y_val, x_test, y_test = asarray_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)
    #plt.imshow(x_train[0], cmap="Greys") 
    #plt.show()
    
    
    if scaling == True:
        pp.standard_scaling([x_train, x_val, x_test])  
    #pp.normalize([x_train, x_val, x_test])
    #plt.imshow(x_train[0], cmap="Greys") 
    #plt.show()
    x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)     

    width = x_train[0].shape[0]
    height = x_train[0].shape[1]
    #batch_size = 40 # standard (5, 12)
    #batch_size = 40
    batch_size = 16 # model 21

    batch_size = 40
    #model_nr = 21

    if model_nr == "4":
        model = create_multifilter_segmentation_model4(width, height)
    elif model_nr == "5": 
        model = create_multifilter_segmentation_model5(width, height)   # standard
    elif model_nr == "6":
        model = create_multifilter_segmentation_model6(width, height)
    elif model_nr == "5a": 
        model = create_multifilter_segmentation_model5a(width, height)
    elif model_nr == "7":
        model = create_multifilter_segmentation_model7(width, height)
    elif model_nr == "8":
        model = create_multifilter_segmentation_model8(width, height) #no laplace, with shape_index and usc
    elif model_nr == "9":
        model = create_multifilter_segmentation_model9(width, height)
    elif model_nr == "10":
        model = create_multifilter_segmentation_model10(width, height)
    elif model_nr == "11":
        model = create_multifilter_segmentation_model11(width, height) # sehr gute Ergebnisse, bei entsprechender Justierung
    elif model_nr == "12":
        model = create_multifilter_segmentation_model12(width, height) # second standard
    elif model_nr == "12_beta":
        model = create_multifilter_segmentation_model12_beta(width, height)
    elif model_nr == "13":
        model = create_multifilter_segmentation_model13(width, height)
    elif model_nr == "14":
        model = create_multifilter_segmentation_model14(width, height)
    elif model_nr == "15":
        model = create_multifilter_segmentation_model15(width, height)
    elif model_nr == "16":
        model = create_multifilter_segmentation_model16(width, height) # x_train:  12697  x_val:  3175  x_test:  7256
    elif model_nr == "17":
        model = create_multifilter_segmentation_model17(width, height)
    elif model_nr == "18":
        model = create_multifilter_segmentation_model18(width, height)
    elif model_nr == "19":
        model = create_multifilter_segmentation_model19(width, height)
    elif model_nr == "20":
        model = create_multifilter_segmentation_model20(width, height)
    elif model_nr == "21":
        model = create_multifilter_segmentation_model21(width, height)
    elif model_nr == "21_tversky":
        model = create_multifilter_segmentation_model21_tversky(width, height)
    elif model_nr == "21_mcc" or model_nr == "21_weighed_tversky":
        model = create_multifilter_segmentation_model21_weighed_tversky(width, height) # bad name. used for mcc training!
    elif model_nr == "23":
        model = create_multifilter_segmentation_model23(width, height)
    elif model_nr == "21_f1_2":
        model = create_multifilter_segmentation_model21_f1_2(width, height)
    elif model_nr == "21_f_beta":
        model = create_multifilter_segmentation_model21_f_beta(width, height)
    elif model_nr == "21_fowlkes_mallows":
        model = create_multifilter_segmentation_model21_fowlkes_mallows(width, height)
    
    model.summary()
    
    epochs = 100
    
     
    timestamp = get_timestamp()
    c_path = "checkpoints/multifilter_segmentation_model_" + str(model_nr) + "_" + timestamp
    checkpoint_path = c_path + "epoch_{epoch:05d}.h5"
    checkpoint_frequency = 20
    callbacks = []
    cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=checkpoint_frequency)
    callbacks.append(cp_callback)

    if lr_decay == True:

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5,
                    patience=3, 
                    min_lr=0.00001)
        
        callbacks.append(reduce_lr)

    
    csv_logger = CSVLogger(c_path + ".log")
    callbacks.append(csv_logger)

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, validation_data=[x_val, y_val])
    score = model.evaluate(x_test, y_test, batch_size=1)

    print("Score: ", score)    
    log_info = "{}, {}, {}\n".format(timestamp, epochs, score)    
    clear_session()
    return log_info



def train_detection_model(model_nr=""): 
    from file_ops import read_h5py_pcr_dataset, read_detection_dataset
    from models import create_multifilter_detection_model
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session  
    from utils import rotate_detection_data  

    #filename = "pcr_dataset_smaller_test_set.h5"
    #filename = "pcr_dataset_without_lower_saxony.h5"
    #filename = "OpenDetectionDataset.h5"
    #filename = "OpenDetectionDataset_2.h5" # open    
    #x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg = read_h5py_pcr_dataset(filename)
    x_train, y_train_class, y_train_reg = read_detection_dataset("D1_train.h5", train=True)
    x_test, y_test_class, y_test_reg = read_detection_dataset("D1_test.h5", test=True)

    for i in range(100): 
        plt.imshow(x_train[i*100]) 
        plt.show()
    sys.exit()


    x_train, y_train_class, y_train_reg = rotate_detection_data(x_train, y_train_class, y_train_reg)
    #   .reshape((-1, 64, 64, 1))

    x_train = x_train.reshape((-1, 64, 64, 1))
    y_train_class = y_train_class.reshape((-1, 64, 64, 1))
    y_train_reg = y_train_reg.reshape((-1, 64, 64, 1))
    x_test = x_test.reshape((-1, 64, 64, 1))
    y_test_class = y_test_class.reshape((-1, 64, 64, 1))
    y_test_reg = y_test_reg.reshape((-1, 64, 64, 1))

    if model_nr == "1":
        model = create_multifilter_detection_model()
    elif model_nr == "2":
        model = create_multifilter_detection_model2()
    elif model_nr == "3":
        model = create_multifilter_detection_model3() # standard
    elif model_nr == "4":
        model = create_multifilter_detection_model4()
    elif model_nr == "5":
        model = create_multifilter_detection_model5()
    elif model_nr == "6":
        model = create_multifilter_detection_model6() # no shape_index
    elif model_nr == "7":
        model = create_multifilter_detection_model7() # bigger segmentation dataset
    elif model_nr == "8":
        model = create_multifilter_detection_model8() # no transfer learning
    elif model_nr == "9":
        model = create_multifilter_detection_model9()
    elif model_nr == "10":
        model = create_multifilter_detection_model10()
    elif model_nr == "11":
        model = create_multifilter_detection_model11() # simpler segmentation model
    elif model_nr == "12":
        model = create_multifilter_detection_model12()
    elif model_nr == "13":
        model = create_multifilter_detection_model13()
    elif model_nr == "14":
        model = create_multifilter_detection_model14(width=64, height=64) # standard (model 2 in MA)
    elif model_nr == "15":
        model = create_multifilter_detection_model15(width=64, height=64) # nabla, laplacian, no rotation
    elif model_nr == "16":
        model = create_multifilter_detection_model16(width=64, height=64) # nabla, laplacian, minimal curvature, mean curvature, no rotation
    elif model_nr == "17":
        model = create_multifilter_detection_model17(width=64, height=64) # nabla, laplacian, minimal curvature, mean curvature, no rotation, with last segmentation layer, (model 3 in MA)
    elif model_nr == "19":
        model = create_multifilter_detection_model19(width=64, height=64)
    elif model_nr == "20":
        model = create_multifilter_detection_model20(width=64, height=64)
    elif model_nr == "21" or model_nr == "":
        model = create_multifilter_detection_model21(width=64, height=64) # open
    elif model_nr == "22":
        model = create_multifilter_detection_model22(width=64, height=64) # halv open
    elif model_nr == "23":
        model = create_multifilter_detection_model23_weighed(width=64, height=64) # f_beta_loss, beta = 0.2
    else: 
        raise Exception("No valid model!")

    #model.load_weights("checkpoints/multifilter_detection_model_23_2021-7-6:18-47-46epoch_00020.h5")
    model.summary()
    #epochs = 100   # standard
    epochs = 100
    batch_size = 32      #32 # standard
    model_nr = 21

    timestamp = get_timestamp()
    c_path = "checkpoints/multifilter_detection_model_" + str(model_nr) + "_" + timestamp
    checkpoint_path = c_path + "epoch_{epoch:05d}.h5"
    checkpoint_frequency = 10
    callbacks = []
    cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=checkpoint_frequency)
    callbacks.append(cp_callback)
    csv_logger = CSVLogger(c_path + ".log")
    callbacks.append(csv_logger)
    
    # only for tversky_loss:
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='classifier_out_loss', 
                factor=0.5,
                patience=3, 
                min_lr=0.00001)
    
    callbacks.append(reduce_lr)   

    model.fit(x_train, [y_train_class, y_train_reg], epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks)
    score = model.evaluate(x_test, [y_test_class, y_test_reg], batch_size=1)

    print("Score: ", score)    
    log_info = "{}, {}, {}\n".format(timestamp, epochs, score)    
    clear_session()
    return log_info


def eval_detection(checkpoint_path, model):

    from file_ops import read_h5py_pcr_dataset
    from models import create_multifilter_detection_model
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session    

    filename = "pcr_dataset_smaller_test_set.h5"
    x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg = read_h5py_pcr_dataset(filename)    

    x_train = x_train.reshape((-1, 64, 64, 1))
    y_train_class = y_train_class.reshape((-1, 64, 64, 1))
    y_train_reg = y_train_reg.reshape((-1, 64, 64, 1))
    x_test = x_test.reshape((-1, 64, 64, 1))
    y_test_class = y_test_class.reshape((-1, 64, 64, 1))
    y_test_reg = y_test_reg.reshape((-1, 64, 64, 1))
    
    model.load_weights(checkpoint_path)
        
    score = model.evaluate(x_test, [y_test_class, y_test_reg], batch_size=1)
    print("Score: ", score)    
    





def eval_segmentation(model, free=True):
    from file_ops import read_h5py_segmentation_dataset
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger    
    from tensorflow.keras.backend import clear_session  
    from utils import rotate_segmentation_data  
        
    #x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius_bigger.h5")
    if free == False:
        x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("frame_segmentation_dataset_smaller_test_set_no_min_radius.h5")
    else: 
        x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("openSegmentationDataset2_big.h5")        
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    
    print("x_train: ", len(x_train), " x_val: ", len(x_val), " x_test: ", len(x_test))
    
    x_test, y_test = shuffle(x_test, y_test)    

    x_train, y_train, x_val, y_val, x_test, y_test = asarray_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)      
    x_train, y_train, x_val, y_val, x_test, y_test = reshape_trainingsset(x_train, y_train, x_val, y_val, x_test, y_test)     
    
    score = model.evaluate(x_test, y_test, batch_size=1)

    print("Score: ", score)    
       
    clear_session()
    





if __name__ == "__main__":
    args = sys.argv[1:]

    if "bigger" in args: 
        bigger = True 
    else: 
        bigger = False


    model_nr = "21" 
    for a in args: 
        if "model=" in a:             
            model_nr = a.split("=")[1] 
            print("model_nr = ", model_nr)
            break

    if "lr_decay" in args: 
        lr_decay = True 
    else: 
        lr_decay = False


    if "evaluation" in args: 

        if "detection" in args: 
            #checkpoint_path = "checkpoints/multifilter_detection/usc/multifilter_detection_model_2021-5-9:0-37-39epoch_00100.h5"
            #model = create_multifilter_detection_model4()

            checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00040.h5" # standard
            #checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00050.h5"  # Score:  [0.19872199130740684, 0.13791345656143555, 0.060808534618596276]
            #checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00060.h5"  # Score:  [0.1904507646370262, 0.13006401551202637, 0.06038674920190622]
            #checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00070.h5"  # Score:  [0.19777389917355442, 0.13321908304408622, 0.06455481592681808]
            #checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00080.h5" # Score:  [0.196184776186439, 0.13322178399790524, 0.06296299243389934]
            #checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00090.h5" # Score:  [0.18704626339335564, 0.12956877854482218, 0.057477484759639215]
            #checkpoint_path = "checkpoints/multifilter_detection_model_15_2021-5-29:9-50-4epoch_00100.h5" # Score:  [0.19012554154211603, 0.12903445779589678, 0.061091084005459304]
            model = create_multifilter_detection_model17(width=64, height=64) #Score:  [0.19749855102351077, 0.13636370159899774, 0.061134849519486334] Samples: 21152
            eval_detection(checkpoint_path, model)

        if "segmentation" in args: 
            #model = create_multifilter_segmentation_model5(64, 64)         
            #checkpoint = "checkpoints/multifilter_detection/multifilter_segmentation_model_2021-5-7:14-21-19epoch_00100.h5"
            #model = create_multifilter_segmentation_model12(64, 64)
            #checkpoint = "checkpoints/multifilter_segmentation_model_2021-5-18:13-35-35epoch_00060.h5"
            #model = create_evaluation_model12(64, 64) # 0.6467796712271795
            #model = create_multifilter_segmentation_model21(64, 64)             
            #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-28:14-45-48epoch_00080.h5") #Score:  [0.19295431296890467, 0.8070456871058443, 0.9944021946277218]
            #model.load_weights("checkpoints/multifilter_segmentation_model_2021-5-28:14-45-48epoch_00100.h5")
            #model.load_weights(checkpoint)
            #model.load_weights("checkpoints/multifilter_segmentation_model_2021-6-1:2-37-59epoch_00100.h5") # bigger set, Score on smaller test set:  [0.35497309111444797, 0.6450269088429712, 0.9918317195467544]
            #model.load_weights("checkpoints/multifilter_segmentation_model_2021-6-1:2-37-59epoch_00060.h5") # bigger set, Score on smaller test set:  [0.371288814915646, 0.6287111850970929, 0.991545453224056]
            #checkpoints = "checkpoints/multifilter_segmentation_model_21_2021-6-7:19-6-28epoch_00060.h5"
            #model.load_weights(checkpoints)

                        
            #model.load_weights("checkpoints/multifilter_segmentation_model_21_2021-6-8:22-15-5epoch_00080_copy.h5") # halv open, Score:  [0.32971512604748804, 0.6702848739299088, 0.9927134732929725]
            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-8:22-15-5epoch_00080.h5"
            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-21:15-44-44epoch_00020.h5" # tversky loss alpha = 0.3 Score:  [0.20187851061583847, 0.776199601424275, 0.993399000121235]
            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-21:15-44-44epoch_00040.h5" # tversky loss alpha = 0.3 Score:  [0.20553547746974352, 0.775719901255698, 0.9929847083366744]
            
            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-21:20-51-10epoch_00020.h5"  # tversky loss alpha = 0.7 Score:  [0.2787904614238533, 0.6912181797155051, 0.9915030321883782]
            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-21:20-51-10epoch_00040.h5"   # tversky loss alpha = 0.7 Score:  [0.28924553604834763, 0.6777762475193344, 0.9920738570734436]
            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-22:2-13-57epoch_00020.h5" # tversky loss alpha = 0.3 with LR decay Score:  [0.22056823548472115, 0.7604690365317792, 0.9938088544769138]
            #model = create_multifilter_segmentation_model21_tversky(64, 64) 

            #checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-6-23:14-46-59epoch_00080.h5" # with learning rate decay # Score:  [0.36204197320027437, 0.6379580268683032, 0.9919442692343674]
            checkpoint = "checkpoints/multifilter_segmentation_model_21_2021-7-1:19-34-14epoch_00100.h5" # mini with rotation Score:  [0.3125084348492696, 0.6874915651087106, 0.991891504848924]
            model = create_multifilter_segmentation_model21(64, 64)

            model.load_weights(checkpoint)

            free = False
            if "open" in args: 
                free = True
            eval_segmentation(model, free)
        sys.exit()

    
    print(args)   
    if "segmentation" in args:
        log_data_single_input = "timestamp, epochs, score\n"


        t = get_timestamp()
        log_file = open("checkpoints/log_data_multiple_filters_single_input_{}.txt".format(t), "w")
        log_file.write(log_data_single_input)
        log_file.close()
               
        log_data_single_input = train_single_input(scaling=False, bigger_test_set=bigger, rotate=True, model_nr=model_nr, lr_decay=lr_decay)      

        log_file = open("checkpoints/log_data_multiple_filters_single_input_{}.txt".format(t), "w")
        log_file.write(log_data_single_input)
        log_file.close()


    if "detection" in args: 
        log_data_single_input = "timestamp, epochs, score\n"


        t = get_timestamp()
        log_file = open("checkpoints/log_data_multiple_filters_detection_{}.txt".format(t), "w")
        log_file.write(log_data_single_input)
        log_file.close()
                    
        log_data_single_input = train_detection_model(model_nr=model_nr)        

        log_file = open("checkpoints/log_data_multiple_filters_detection_{}.txt".format(t), "w")
        log_file.write(log_data_single_input)
        log_file.close()

    
        