import glob
import numpy as np 
import matplotlib.pyplot as plt 
import file_ops   
from bounding_box import BBox
from vis import slrm
import preprocessing as pp


def save_rgb_image_from_numpy_matrices(filename, matrices): 
    import cv2
    # matrices: one 2d-array for every colour channel. 
    for i in range(3): 
        matrices[i] -= np.min(matrices[i])
        matrices[i] /= np.max(np.abs(matrices[i]))     


    if type(matrices) == list: 
        matrices = np.asarray(matrices) 
    matrices *= 255.
    matrices = cv2.merge(matrices)
    #matrices = matrices.reshape((matrices.shape[1], matrices.shape[2], matrices.shape[0]))
    #print(matrices.shape)
    cv2.imwrite(filename, matrices)
    #cv2.imshow("image", img)
    #cv2.waitKey() 





def convert_color(matrix): 
    matrix *= (-1)
    matrix -= np.min(matrix)
    matrix /= np.max(matrix)
    matrix *= 256      
    return matrix 


def convert_and_write(matrix, name, path="", convert=True): 
    # Save a 2d-numpy-array with cv2 as png image. 
    from cv2 import imwrite
    if convert == True:
        matrix = convert_color(matrix)
    #m = np.copy(matrix) 
    #m -= np.min(m) 
    #m /= np.max(m)
    imwrite(path + "{}.png".format(name), matrix)



def show_image_with_bboxes(image, bbox_list):
    from cv2 import rectangle, imshow
    color = (255, 0, 0)
    thickness = 1
    for bbox in bbox_list: 
        xmin = bbox.xmin 
        xmax = bbox.xmax 
        ymin = bbox.ymin 
        ymax = bbox.ymax 
        rectangle(image,(xmin, ymin), (xmax, ymax), color, thickness)
    """
    z = np.zeros((64, 64))
    for i in range(64): 
        for j in range(64): 
            z[i][j] = np.max(image[i][j])
    image = z
    """
    plt.imshow(image) 
    plt.show()



def draw_bounding_circle(matrix, x, y, radius, v): 
    # draws a bounding circle around middle point at (x, y) and given radius. 
    # value for filling is v.
    from math import sqrt
    r = round(radius) 
    for i in range(matrix.shape[0]): 
        for j in range(matrix.shape[1]): 
            d = abs(i - x)**2 + abs(j - y)**2  
            d = sqrt(d) 
            d = round(d)
            if d == r: 
                matrix[i, j] = v 
    return matrix 


def add_bounding_circles_from_pcr(m, c, r, v, threshold=0.5): 
    # m: original frame 
    # c: predicted class matrix 
    # r: predicted radius matrix 
    # v: value for bounding circle color    
    for i in range(m.shape[0]): 
        for j in range(m.shape[1]):
            if c[i, j] > threshold: 
                m = draw_bounding_circle(m, i, j, r[i, j], v) 
    return m


def bbox_management(m, c, r, v, threshold=0.5, tolerance=1., circle_value=1, return_circle_list=False): 
    import operator
    from math import sqrt
    width, height = m.shape
    circle_matrix = np.zeros(m.shape)
    circle_list = []
    for i in range(width): 
        for j in range(height): 
            if c[i, j] >= threshold:  
                #circle_matrix = draw_bounding_circle(circle_matrix, i, j, r[i, j], 1)
                circle_list.append([i, j, c[i, j], r[i, j]])
    circle_list.sort(key=operator.itemgetter(2), reverse=True) 
    #print(circle_list)
    for i in range(len(circle_list)):        
        for j in range(len(circle_list) - 1, i, -1): 
            #print(i, j)
            x1 = circle_list[i][0] 
            x2 = circle_list[j][0] 
            y1 = circle_list[i][1] 
            y2 = circle_list[j][1]
            r1 = circle_list[i][3]
            r2 = circle_list[j][3]
            c1 = circle_list[i][2] 
            c2 = circle_list[j][2]
            D = sqrt((x2 - x1)**2 + (y2 - y1)**2) 
            if (D < tolerance * r1 or D < tolerance * r2): #c1 >= c2 and 
                #print("Should delete!")
                del circle_list[j]
            #else: 
            #    pass 
            #    print(D, r1, r2)

    mean_radius = 0 
    
    for c in circle_list: 
        circle_matrix = draw_bounding_circle(circle_matrix, c[0], c[1], c[3], circle_value) 
        mean_radius += c[-1]
    if len(circle_list) > 0:
        mean_radius /= len(circle_list)
    print("mean radius (in meters): ", mean_radius * 2) # the radii are in DTM2-format, so multiplying by 2 is neccessary to get the true value. 
    print("Number of circles: ", len(circle_list))
    if return_circle_list == False:
        return circle_matrix
    else: 
        return circle_matrix, circle_list


            

def bbox_management2(m, c, r, v, threshold=0.5, tolerance=1.): 
    import operator
    from math import sqrt
    width, height = m.shape
    circle_matrix = np.zeros(m.shape)
    circle_list = []
    for i in range(width): 
        for j in range(height): 
            if c[i, j] >= threshold:  
                #circle_matrix = draw_bounding_circle(circle_matrix, i, j, r[i, j], 1)
                circle_list.append([i, j, c[i, j], r[i, j]])
    circle_list.sort(key=operator.itemgetter(2), reverse=True) 
    print(circle_list)
    for i in range(len(circle_list)):        
        for j in range(len(circle_list) - 1, i, -1): 
            #print(i, j)
            x1 = circle_list[i][0] 
            x2 = circle_list[j][0] 
            y1 = circle_list[i][1] 
            y2 = circle_list[j][1]
            r1 = circle_list[i][3]
            r2 = circle_list[j][3]
            c1 = circle_list[i][2] 
            c2 = circle_list[j][2]
            D = sqrt((x2 - x1)**2 + (y2 - y1)**2) 
            if D < (tolerance * (r1 + r2)):                 
                del circle_list[j]            

    mean_radius = 0 
    
    for c in circle_list: 
        circle_matrix = draw_bounding_circle(circle_matrix, c[0], c[1], c[3], 1) 
        mean_radius += c[-1]
    if len(circle_list) > 0:
        mean_radius /= len(circle_list)
    print("mean radius (in meters): ", mean_radius * 2) # the radii are in DTM2-format, so multiplying by 2 is neccessary to get the true value. 
    print("Number of circles: ", len(circle_list))
    return circle_matrix





def trainingsset_to_png_and_npy(train, img_path, npy_path): 
    # img_path: "images/trainingsset/img_files/"
    # npy_path: "images/trainingsset/npy_files/"
    from preprocessing import write_shade_as_png 
    for t in range(len(train)) : 
        write_shade_as_png(train[t], img_path + ":" + str(i) + ".png") 
        np.save(npy_path + ":" + str(i) + ".npy", train[t]) 


def png_and_npy_to_trainingsset(img_path, npy_path): 
    # img_path: "images/trainingsset/img_files/"
    # npy_path: "images/trainingsset/npy_files/"
    import glob  
    train = []
    for filename in glob.glob(img_path): 
        filename = filename.replace(".png", "") 
        f = filename.split(":")[-1]
        filename = int(f)
        matrix = np.load(npy_path + ":" + f + ".npy") 
        train.append(matrix) 
    return np.asarray(train) 



def rotate_segmentation_data(x_train, y_train): 
    from random import randint 
    for i in range(len(x_train)):
        if i % 2 == 1:             
            r = randint(0, 3) 
            #print(r)
            #plt.imshow(x_train[i])
            #plt.show()
            x_train[i] = np.rot90(x_train[i], k=r) 
            y_train[i] = np.rot90(y_train[i], k=r)    
            #plt.imshow(x_train[i])
            #plt.show()         
    return x_train, y_train


def rotate_detection_data(x_train, y_train_class, y_train_reg): 
    from random import randint 
    for i in range(len(x_train)):
        if i % 2 == 1: 
            r = randint(0, 3) 
            x_train[i] = np.rot90(x_train[i], k=r) 
            y_train_class[i] = np.rot90(y_train_class[i], k=r) 
            y_train_reg[i] = np.rot90(y_train_reg[i], k=r) 
    return x_train, y_train_class, y_train_reg




def split_matrix(matrix, slice_size):
    # splits a big matrix into smaller ones. 
    # matrix: the matrix to be splitted, 
    # slice_size: The desired height and width of the partial matrices.
    # Returns: - a dictionary of the partial matrices with the positions 
    #            as key tuples (x,y), 
    #          - a tuple with the maximum x and y values (x_max, y_max). 
    from numpy import array_split 

    slice_nr_x = matrix.shape[0] % slice_size 
    slice_nr_y = matrix.shape[1] % slice_size 
    slices_x = array_split(matrix, slice_nr_x) 
    slices = []
    slices_dict = {}

    for s in slices_x: 
        slices_y = array_split(s, slice_nr_y, axis=1)
        slices.append(slices_y)

    for x in range(len(slices)) : 
        for y in range(len(slices[x])): 
            slices_dict[(x,y)] = slices[x][y]

    return slices_dict, (len(slices), len(slices[x]))


def merge_matrix(m_dict, m_size=(-1, -1)):   
    # rebuilds a bigger matrix that has been splitted by split_matrix() 
    # from the dictionary that has been returned from this function and 
    # (optionally) the size tuple of that dictionary.  
    # Returns: A matrix that should resemble the input matrix of split_matrix().
    if m_size == (-1, -1): 
        x_max = 0 
        y_max = 0 
        keys = m_dict.keys() 
        for key in keys: 
            if key[0] > x_max: 
                x_max = key[0] 
            if key[1] > y_max: 
                y_max = key[1] 
        m_size = (x_max, y_max)

    slices = []   
    for x in range(m_size[0]): 
        s = m_dict[(x, 0)]
        for y in range(1, m_size[1]): 
            s = np.concatenate((s, m_dict[x, y]), axis=1) 
        slices.append(s) 
    slices = tuple(slices)
    return np.concatenate(slices, axis=0)





def get_four_rotation_matrices(matrix, return_type="", zero_at_front=False): 
    # return_type can be set to "list" instead of individual matrices. 
    # A list would be more practical, but there are functions implemented 
    # using the individual version. 
    # zero_at_front defines if the original matrix is placed at the front or 
    # the back. 

    matrix1 = np.rot90(matrix, k=1)
    matrix2 = np.rot90(matrix, k=2)
    matrix3 = np.rot90(matrix, k=3)
    matrix0 = np.copy(matrix) 
    if zero_at_front == False: 
        if return_type == "list": 
            return [matrix1, matrix2, matrix3, matrix0]
        else: 
            return matrix1, matrix2, matrix3, matrix0
    else: 
        if return_type == "list": 
            return [matrix0, matrix1, matrix2, matrix3]
        else: 
            return matrix0, matrix1, matrix2, matrix3 



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


def get_timestamp():
    import time
    x = time.localtime() 
    return str(x[0]) + "-" + str(x[1]) + "-" + str(x[2]) + ":" + str(x[3]) + "-" + str(x[4]) + "-" + str(x[5]) 



def get_window_by_position(x_size, y_size, matrix, x_start, y_start):
    # copies a rectangular shaped area out of the matrix and returns it together with 
    # its bounding box.
    x_end = x_start + x_size
    y_end = y_start + y_size
    b_box = [x_start, x_end, y_start, y_end]
    window = matrix[x_start:x_end, y_start:y_end].copy()
    #print(matrix.shape, x_start, x_end, y_start, y_end)
    return [window, b_box]


def get_crxy_window_by_position(x_size, y_size, matrices, x_start, y_start):
    # copies a rectangular shaped area out of the matrix and returns it together with 
    # its bounding box.
    x_end = x_start + x_size
    y_end = y_start + y_size
    b_box = [x_start, x_end, y_start, y_end]
    windows = []
    for matrix in matrices:
        window = matrix[x_start:x_end, y_start:y_end] 
        windows.append(window) 
    windows = np.asarray(windows)
    #print(matrix.shape, x_start, x_end, y_start, y_end)
    return [windows, b_box]


def get_window_list_by_position(matrices, bbox_list):
    window_list = []
    for matrix in matrices: 
        for box in bbox_list:
            x_size = box.width 
            y_size = box.height
            box.print()
            window = get_window_by_position(y_size, x_size, matrix, box.ymin, box.xmin)[0]
            window -= np.min(window)               
            window_list.append(window)     
    return window_list


def matrix_smallsizing(augmentation_steps):
    smallsized_matrices = []            
    matrices = smallsize_matrix_general(self.matrix, augmentation_steps)          
    return matrices



def get_frame_with_bbox_info(   matrix, 
                                bbox_list, 
                                frame_xmin, 
                                frame_ymin, 
                                frame_ID, 
                                filepath="", 
                                kernel_size=64, 
                                file_type="npy", 
                                augmentation=False, 
                                augmentation_steps=-1   ):
    # saves file as numpy array per default, not as image! 
    
    from augment_by_smallsizing import smallsize_matrix_general
    bb_list = bbox_list.copy()
    #for b in range(len(bb_list)): 
        #print("BBox b: ", bbox_list[b].xmin, bbox_list[b].ymin, bbox_list[b].xmax, bbox_list[b].ymax)
    frame_xmax = frame_xmin + kernel_size - 1 
    frame_ymax = frame_ymin + kernel_size - 1    

    frame = get_window_by_position(kernel_size, kernel_size, matrix, frame_ymin, frame_xmin)[0]
    #frame_bbox_list = [] 
    csv_lines = ""
    filename = filepath.replace(".png", "") + "_" + str(frame_ID) + "." + file_type
    #print("Frame", frame_xmin, frame_ymin)
    if augmentation: 
        augmentation_factor = augmentation_steps 
        frames = smallsize_matrix_general(frame, augmentation_steps)
        new_kernel_size = int(kernel_size / augmentation_steps) 

        frame_xmin2 = int(frame_xmin / augmentation_steps)  
        frame_ymin2 = int(frame_ymin / augmentation_steps) 
        frame_xmax2 = int(frame_xmax / augmentation_steps)
        frame_ymax2 = int(frame_ymax / augmentation_steps)
        
    else: 
        augmentation_factor = 1
        frames = [frame]
        new_kernel_size = kernel_size            
    index = 0 
    #print(frames[0])
    for frame in frames:   
        #print(frame.shape, new_kernel_size)            
        frame_filename = filename.replace("." + file_type , "_" + str(index) + "." + file_type)      
        for bbox in bb_list:
            
            
            bbox_xmin = max(bbox.xmin - frame_xmin2, 0) 
            bbox_ymin = max(bbox.ymin - frame_ymin2, 0)
            bbox_xmax = min(bbox.xmax - frame_xmin2, frame_xmax2) 
            bbox_ymax = min(bbox.ymax - frame_ymin2, frame_ymax2) 
            #print("Frame", frame_xmin, frame_ymin, ": ", bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
            #print("BBox: ", bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)
            bbox_width = bbox_xmax - bbox_xmin 
            bbox_height = bbox_ymax - bbox_ymin 
            # and max([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]) < 64 and min([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]) >= 0
            if bbox_width > 3 and bbox_height > 3 and bbox.label == "gravemound" and max([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]) < new_kernel_size and min([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]) >= 0:   
                #index += 1                  
                csv_line = frame_filename + "," + str(frame.shape[0]) + "," + str(frame.shape[1]) + "," + bbox.label + "," + str(bbox_xmin) + "," + str(bbox_ymin) + "," + str(bbox_xmax) + "," + str(bbox_ymax) + "\n" 
                csv_line = csv_line.replace("." + file_type, "." + "png")
                print("CSV Line: ", csv_line)
                #print(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
                csv_lines += csv_line
        if csv_lines != "": 
            np.save(frame_filename, frame) 
        index += 1
    return csv_lines


def get_pcr_frame_and_label(    matrix, 
                                bbox_list, 
                                frame_xmin, 
                                frame_ymin,                                 
                                kernel_size=64,                                 
                                augmentation=False, 
                                augmentation_steps=-1   ):
    # pcr: pixelwise classification and regression
    from augment_by_smallsizing import smallsize_matrix_general
    from preprocessing import get_gradients, laplacian
    #print("augmentation: ", augmentation) 
    #print("augmentation_steps: ", augmentation_steps)
    bb_list = bbox_list.copy()    
    frame_xmax = frame_xmin + kernel_size - 1 
    frame_ymax = frame_ymin + kernel_size - 1    

    frame = get_window_by_position(kernel_size, kernel_size, matrix, frame_ymin, frame_xmin)[0]
    #print("Frame before smallsizing: ", frame.shape)   
    input_frames = [] 
    output_classes = [] 
    output_radii = []     
    
    if augmentation: 
        augmentation_factor = augmentation_steps 
        frames = smallsize_matrix_general(frame, augmentation_steps)
        #print("Frame after smallsizing: ", frames[0].shape) 
        new_kernel_size = int(kernel_size / augmentation_steps) 
        c_frames = [np.zeros((new_kernel_size, new_kernel_size)) for i in range(len(frames))] 
        r_frames = [np.zeros((new_kernel_size, new_kernel_size)) for i in range(len(frames))]
        frame_xmin2 = int(frame_xmin / augmentation_steps)  
        frame_ymin2 = int(frame_ymin / augmentation_steps) 
        frame_xmax2 = int(frame_xmax / augmentation_steps)
        frame_ymax2 = int(frame_ymax / augmentation_steps)
        
    else: 
        augmentation_factor = 1
        frames = [frame]
        new_kernel_size = kernel_size  
        c_frames = [np.zeros((new_kernel_size, new_kernel_size))] 
        r_frames = [np.zeros((new_kernel_size, new_kernel_size))] 
        frame_xmin2 = frame_xmin 
        frame_ymin2 = frame_ymin
        frame_xmax2 = frame_xmax 
        frame_ymax2 = frame_ymax 
     
    for i in range(len(frames)): 
        #print(new_kernel_size, frames[i].shape)
        if frames[i].shape == (new_kernel_size, new_kernel_size):              
            for bbox in bb_list:             
                a = (bbox.xmax - bbox.xmin) / 2 
                b = bbox.xmin - frame_xmin2 
                y_middle = int(b + a) 

                a = (bbox.ymax - bbox.ymin) / 2 
                b = bbox.ymin - frame_ymin2 
                x_middle = int(b + a)              
                        
                radius = (bbox.width + bbox.height) / 2         
                if min(x_middle, y_middle) >= 0 and max(x_middle, y_middle) < new_kernel_size: 
                    c_frames[i][x_middle][y_middle] = 1 
                    r_frames[i][x_middle][y_middle] = radius
            
            if np.sum(c_frames[i]) > 2 and augmentation_steps==-2:
                p, q, r, s, t = get_gradients(frames[i])
                #frames[i] = slope(p,q)
                frames[i] = laplacian(r, t)
                frames[i] = add_bounding_circles_from_pcr(frames[i], c_frames[i], r_frames[i], 0)
                plt.subplot(131)
                plt.imshow(np.clip(frames[i], -1, 1), cmap="Greys") 
                plt.subplot(132)
                #plt.show() 
                plt.imshow(c_frames[i])
                #plt.show() 
                plt.subplot(133)
                plt.imshow(r_frames[i]) 
                plt.show()
            
            input_frames.append(frames[i]) 
            output_classes.append(c_frames[i]) 
            output_radii.append(r_frames[i])             
    #print("Len Input Frames: ", len(input_frames))    
    return input_frames, output_classes, output_radii




def get_matrix_with_bbox_info(  matrix, 
                                bbox_list,                                 
                                filepath="", 
                                file_type="npy", 
                                augmentation=False, 
                                augmentation_steps=-1   ):
    # saves file as numpy array per default, not as image!     
    from augment_by_smallsizing import smallsize_matrix_general

    bb_list = bbox_list.copy()
    frame_xmin = 0 
    frame_ymin = 0
    frame_xmax = matrix.shape[0] - 1 
    frame_ymax = matrix.shape[1] - 1    

    frame = matrix.copy()
    csv_lines = ""
    filename = filepath.replace(".png", "." + file_type)  
    if augmentation: 
        augmentation_factor = augmentation_steps 
        frames = smallsize_matrix_general(frame, augmentation_steps)
        

        frame_xmin2 = 0  
        frame_ymin2 = 0 
        frame_xmax2 = int(frame_xmax / augmentation_steps)
        frame_ymax2 = int(frame_ymax / augmentation_steps)        
    else: 
        augmentation_factor = 1
        frames = [frame]
             
    index = 0 
    for frame in frames:            
        frame_filename = filename.replace("." + file_type , "_" + str(index) + "." + file_type)  

        for bbox in bb_list:                   
            bbox_xmin = max(bbox.xmin - frame_xmin2, 0) 
            bbox_ymin = max(bbox.ymin - frame_ymin2, 0)
            bbox_xmax = min(bbox.xmax - frame_xmin2, frame_xmax2) 
            bbox_ymax = min(bbox.ymax - frame_ymin2, frame_ymax2) 

            bbox_width = bbox_xmax - bbox_xmin 
            bbox_height = bbox_ymax - bbox_ymin 
            if bbox_width > 3 and bbox_height > 3 and bbox.label == "gravemound":                                
                csv_line = frame_filename + "," + str(frame.shape[0]) + "," + str(frame.shape[1]) + "," + bbox.label + "," + str(bbox_xmin) + "," + str(bbox_ymin) + "," + str(bbox_xmax) + "," + str(bbox_ymax) + "\n" 
                csv_line = csv_line.replace("." + file_type, "." + "png")
                print("CSV Line: ", csv_line)
                csv_lines += csv_line
        if csv_lines != "": 
            np.save(frame_filename, frame) 
        index += 1
    return csv_lines



def manipulate_label_matrix(matrix): 
    #matrix = np.load(filename) 
    for i in range(matrix.shape[0]): 
        for j in range(matrix.shape[1]):
            if matrix[i][j] < 0.9:
                matrix[i][j] = 0.
    return matrix

def print_prediction_matrices(matrix, label_matrix): 
    plt.subplot(1,2,1) 
    plt.imshow(matrix) 
    plt.subplot(1,2,2) 
    plt.imshow(label_matrix) 
    plt.show()


def print_all_prediction_matrices():     
    for filename in glob.glob("images/test_data/" + "/*.npy"):        
        data = np.load(filename)
        matrix = data[0] 
        label_matrix = data[1]
        manipulate_label_matrix(label_matrix)
        print_prediction_matrices(matrix, label_matrix)

def print_one_prediction_matrix():
    filename = "images/test_data/333665718_dgm2.npy"    
    data = np.load(filename)
    matrix = data[0] 
    label_matrix = data[1]
    matrix = np.rot90(matrix)
    label_matrix = np.rot90(label_matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 0: 
                matrix[i][j] = 79 
    #file_ops.write_ply_file(matrix, matrix.shape[0], matrix.shape[1], "images/Rosenfeld1")
    manipulate_label_matrix(label_matrix)
    print_prediction_matrices(matrix, matrix + 10 * label_matrix)

def check_dataset_for_flipping(): 
    from sklearn.model_selection import train_test_split
    data = np.load("segmentation_frames_smallsized4.npy")
    X = data[0] 
    Y = data[1]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4) 
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5) 

    x_train_flipped = np.flip(x_train, axis=2) 
    plt.subplot(1,2,1) 
    plt.imshow(x_train[0]) 
    plt.subplot(1,2,2) 
    plt.imshow(x_train_flipped[0]) 
    plt.show()


def check_dataset_data(filename): 
    from file_ops import read_h5py_segmentation_dataset
    x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset(filename)
    for i in [x_train, y_train, x_test, y_test]: 
        print(i.shape)



def shift(matrix, x, y): 
    from cv2 import warpAffine
    t = [[1., 0., x], [0., 1., y]] 
    t = np.asarray(t) 
    shifted = warpAffine(matrix, t, (matrix.shape[1], matrix.shape[0])) 
    return shifted 

def test_shift(): 
    from preprocessing import get_gradients, slope 
    
    matrix = np.load("images/vieritz.npy") 
    p, q, r, s, t = get_gradients(matrix) 
    matrix = slope(p, q)
    shifted = shift(matrix, -100, -100)
    plt.imshow(shifted) 
    plt.show()





def save_filter_image(filepath, method): 
    import preprocessing as pp
    img = create_filter_example_multiscale() 
    img = pp.transform(img, method)
    convert_and_write(img, method, path=filepath)

def create_filter_images(): 
    img_path = "images/filter_examples/"
    
    methods = [
        "slrm_cv2_average",
        "slrm_cv2_gaussian",
        "slrm_cv2_median",
        "slrm_cv2_bilateral",
        "cv2_laplacian",
        "cv2_sobel2D",
        "double_sobel",
        "double_slope",
        "double_slope_one_dir",
        "slope",  
        "aspect",
        "northwardness",
        "eastwardness",
        "plan_curvature",
        "horizontal_curvature",
        "vertical_curvature",
        "difference_curvature",
        "accumulation_curvature",  
        "ring_curvature",
        "rotor",
        "horizontal_curvature_deflection",
        "vertical_curvature_deflection",
        "mean_curvature",
        "gaussian_curvature",
        "minimal_curvature",
        "maximal_curvature",
        "unsphericity_curvature",
        "horizontal_excess_curvature",
        "vertical_excess_curvature", 
        "laplacian", 
        "shape_index",
        "curvedness",
        "reflectance",
        "insolation",
        "pseudo_slope",
        "sky_view_factor", 
        "sky_illumination",
        "local_dominance"
    ]
    
    matrix = create_filter_example_multiscale()
    convert_and_write(matrix, "original", path=img_path)
    for method in methods: 
        try: 
            save_filter_image(img_path, method) 
        except: 
            print("create_filter_images: Error at method {}!".format(method))



def create_trainingsset_with_different_minima(min_v=9, max_v=16): 
    # min_v inclusive, 
    # max_v exclusive!
    from file_ops import create_compressed_h5py_segmentation_dataset
    for i in range(min_v, max_v): 
        filename = "comp_seg_data_width-" + str(i) + ".h5"
        print(filename)
        create_compressed_h5py_segmentation_dataset(filename, min_width=i)
        

def create_trainingsset_without_mani(): 
    from file_ops import create_compressed_h5py_segmentation_dataset
    create_compressed_h5py_segmentation_dataset("segmentation_dataset_no_bbox_mani.h5", bbox_manipulation=0, min_width=8)




def show_log_data(filename):     
    import pandas as pd 
    rt = pd.read_csv(filename) 
    t_loss = rt["loss"].to_numpy() 
    v_loss = rt["val_loss"].to_numpy() 
    x = rt["epoch"].to_numpy()
    plt.plot(x, t_loss, color="r", label="train-loss") 
    plt.plot(x, v_loss, color="b", label="val-loss") 
    plt.legend(loc="upper right")
    plt.show()



def show_multifilter_detection_log_data(filename, usc=True):     
    import pandas as pd 
    rt = pd.read_csv(filename)     
    x = rt["epoch"].to_numpy()
    if usc == True:
        c = rt["conv2d_22_loss"].to_numpy()
        r = rt["conv2d_24_loss"].to_numpy()
    else: 
        raise Exception("Error: Only USC-option implemented! Set usc == True!")   
    plt.plot(x, c, color="r", label="classification-loss") 
    plt.plot(x, r, color="b", label="regression-loss")

    plt.legend(loc="upper right")
    plt.show()







def remove_outliers_from_matrix(matrix, threshold, mean_type="median"): 
    # removes values that are to big or to small to be a realistic part of the DTM
    if mean_type == "median": 
        median = np.median(matrix)
        upper = median + threshold 
        lower = median - threshold 
    return np.clip(matrix, lower, upper)
 




def remove_nan(tensor, filler=0.000000001): 
    import tensorflow as tf   
    try: 
        return tf.where(tf.is_nan(tensor), tf.zeros_like(tensor) + tf.constant(filler), tensor)
    except: 
        return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor) + tf.constant(filler), tensor)

    

def show_filter(m_filter="horizontal_curvature_deflection"): 
    m = np.load("images/vieritz.npy") 
    m = pp.transform(m, m_filter)
    m = np.clip(m, -0.05, 0.05)
    plt.imshow(m, cmap="Greys")
    plt.show()




def get_rgb_color(heigth):
    x = heigth + 128
    R = 0
    G = 0
    B = x % 255
    if x % 255 == 0 and x != 0:
        B = 255

    if x <= 255:
        return [R, G, B]
    elif x > 255 and x < 511:
        G = B
        B = abs(255 - B)            
        return [R, G, B]            
    elif x >= 511 and x < 766:
        G = 255
        R = B
        B = 0        
        return [R, G, B]   
    elif x >= 766 and x < 1021:
        G = abs(255 - B)
        R = 255    
        B = 0    
        return [R, G, B]
    else:
        G = B
        R = 255    
        B = B
        return [R, G, B]   
