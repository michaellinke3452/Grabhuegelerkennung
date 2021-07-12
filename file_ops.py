import math as m
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit



def xml_to_bbox(xmlpath, voc=False):
    # reads bounding-box information from XML file as it is made by 
    # labelImg and returns the information as a list of BBox objects.
    if type(xmlpath) != str: 
        raise TypeError("xml_to_bbox: xmlpath must be a string!")
    elif ".xml" not in xmltype: 
        raise Exception("xml_to_bbox: Not an xml path!")
    from bounding_box import BBox 
    import glob
    import xml.etree.ElementTree as ET
    import pandas as pd

    xml_list = []
    ID = 1
    bbox_list = []

    for xml_file in glob.glob(xmlpath):     #path + '/*.xml'
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            fn = root.find('filename').text.split(".")[0]
            #w = int(root.find('size')[0].text)
            #h = int(root.find('size')[1].text)
            c = member[0].text

            if voc == False: 
                xmin = int(member[4][0].text)
                ymin = int(member[4][1].text)
                xmax = int(member[4][2].text)
                ymax = int(member[4][3].text)
            else:  
                xmin = int(member[4][0].text)
                xmax = int(member[4][1].text)
                ymin = int(member[4][2].text)
                ymax = int(member[4][3].text)

            bbox = BBox(xmin, xmax, ymin, ymax, fn, c, ID, is_voc=voc)
            bbox_list.append(bbox)
            ID += 1

    return bbox_list



def geotif_to_matrix(filepath, lower=-10000): 
    # opens a geotif file and converts it into a numpy array. 
    # returns a numpy array.
    if type(filepath) != str: 
        raise TypeError("geotif_to_matrix: filepath must be a string!")
    import gdal 
    data = gdal.Open(filepath)
    matrix = np.array(data.GetRasterBand(1).ReadAsArray())
    min_value = 100000000.
    for i in range(matrix.shape[0]): 
        for j in range(matrix.shape[1]): 
            v = matrix[i][j]
            if v < min_value and v > lower: 
                min_value = v 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i][j]
            if v < lower:
                matrix[i][j] = min_value
    matrix -= min_value    
    return matrix


def write_bb_csv_file(bb_list, filename):
    # saves a list of bounding box coordinates in a csv file.
    data = open((filename + ".csv"), "w")
    for i in bb_list:
        for j in range(0, 3):
            data.write(str(i[j]))
            data.write(";")
        data.write(str(i[3]))
        data.write("\n")
    data.close()

def read_bb_csv_file(filename):
    # reads bounding box information from csv file and returns it as 
    # a list of coordinates.
    bb_file = open(filename + ".csv")
    data = bb_file.read()
    bb_file.close()
    lines = data.split("\n")
    boxes = []
    for i in lines:
        try:
            columns = i.split(";") 
            boxes.append([int(columns[0]), int(columns[1]), int(columns[2]), int(columns[3])])
        except:
            pass
    return boxes

"""
def read_bb_xml_file_voc(filename):
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    #                    0         1         2        3        4       5      6        7
    #                                                          4       6      5        7
    b_boxes = [] 
"""

def read_bb_csv_file_voc(filename, rotate="no_rotation"):
    # reads bounding box information from csv file in VOC order and 
    # returns it as a list of coordinates.
    if rotate != "no_rotation" and rotate != "rotation":
        print("Function read_bb_csv_file_voc: No valid rotate variable set!\nOption no_rotation chosen!")
    bb_file = open(filename + ".csv")
    data = bb_file.read()
    #print(data)
    bb_file.close()
    lines = data.split("\n")
    boxes = []
    for i in lines:
        try:
            columns = i.split(",") 
            #if rotate == "no_rotation":
            #    boxes.append([int(columns[4]), int(columns[6]), int(columns[5]), int(columns[7])])
            if rotate == "rotation":
                boxes.append([int(columns[5]), int(columns[7]), int(columns[4]), int(columns[6])])
            else:
                boxes.append([int(columns[4]), int(columns[6]), int(columns[5]), int(columns[7])])
                
        except:
            pass
    return boxes


def write_xyz_file(matrix, matrix_size, filename):
    # writes the data of a 2D array into an XYZ file. 
    # the matrix represented by the 2D array is square. 
    # it requires the XYZ file to be stored in a subfolder called "data".
    # matrix: 2D numpy array, 
    # matrix_size width and height of the matrix (one integer), 
    # filename: string, only the filename without ending.
    if not type(matrix) is np.ndarray: 
        raise TypeError("write_xyz_file: matrix must be a numpy.ndarray!")
    if type(matrix_size) != int: 
        raise TypeError("write_xyz_file: matrix_size must be an integer!") 
    if type(filename) != str: 
        raise TypeError("write_xyz_file: filename must be a string!")

    x_max = matrix_size
    y_max = matrix_size
    xyz_data = open(("data/" + filename + ".xyz"), "w")
    for x in range(0, x_max):
        for y in range(0, y_max):
            xyz_data.write(str(x))
            xyz_data.write(" ")
            xyz_data.write(str(y))
            xyz_data.write(" ")
            xyz_data.write(str(matrix[x][y]))
            xyz_data.write("\n")
    xyz_data.close()


def write_xyz_file(matrix, x_max, y_max, filename):
    # writes the data of a 2D array into an XYZ file. 
    # matrix: 2D numpy array, 
    # x_max, y_max: width and height of the matrix, both integers, 
    # filename: string, path + filename without ending.
    if not type(matrix) is np.ndarray: 
        raise TypeError("write_xyz_file: matrix must be a numpy.ndarray!")
    if type(x_max) != int or type(y_max) != int: 
        raise TypeError("write_xyz_file: x_max and y_max must be integers!") 
    if type(filename) != str: 
        raise TypeError("write_xyz_file: filename must be a string!")

    print((filename  + ".xyz"))
    xyz_data = open((filename + ".xyz"), "w")
    for x in range(0, x_max):
        for y in range(0, y_max):
            xyz_data.write(str(x))
            xyz_data.write(" ")
            xyz_data.write(str(y))
            xyz_data.write(" ")
            xyz_data.write(str(matrix[x][y]))
            xyz_data.write("\n")
    xyz_data.close()


def get_xyz_data_from_file(filename):
    # filename: path to XYZ file with ending.
    # returns a list of xyz-coordinates.

    if type(filename) != str: 
        raise TypeError("get_xyz_data_from_file: filename must be a string!")
    d = open(filename)
    inhalt = d.read()
    d.close()
    zeilen = inhalt.split("\n")
    xyz_data = []
    for i in zeilen:    
        string_data = i.split(" ")
        #print(string_data)
        if len(string_data) == 3:
            #print([string_data[0], string_data[1], string_data[2]])
            xyz_data.append([float(string_data[0]), float(string_data[1]), float(string_data[2])])
    return xyz_data


def get_min_and_max_from_xyz_file(filename):
    # returns the highest values in each column of an XYZ file.
    if type(filename) != str: 
        raise TypeError("get_min_and_max_from_xyz_file: filename must be a string!")

    with open(filename) as d:
        inhalt = d.read()    
    zeilen = inhalt.split("\n")    
    x = [] 
    y = []
    for i in zeilen:         
        string_data = i.split(" ")        
        if len(string_data) in [3, 6]:
            try:                
                x.append(float(string_data[0]))
                y.append(float(string_data[1]))
            except:                 
                pass
    print("Lengths: ", len(x), len(y))
    min_x = min(x) 
    max_x = max(x) 
    min_y = min(y)
    max_y = max(y) 
    
    return min_x, max_x, min_y, max_y


def ply_header(nr_of_vertices):
    # returns the head of a PLY file. 
    if type(nr_of_vertices) not in [int, str]: 
        raise TypeError("ply_header: nr_of_vertices must be an integer or string!")

    header = "ply\nformat ascii 1.0\nelement vertex " + str(nr_of_vertices) + "\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float r\nproperty float g\nproperty float b\n"
    header += "end_header\n"
    print(header)
    return header


def write_ply_file(matrix, matrix_size, filename):
    # saves a square 2D numpy-array as PLY file.
    if not type(matrix) is np.ndarray: 
        raise TypeError("write_ply_file: matrix must be a numpy.ndarray!")
    if type(matrix_size) != int: 
        raise TypeError("write_ply_file: matrix_size must be an integer!") 
    if type(filename) != str: 
        raise TypeError("write_ply_file: filename must be a string!")

    from utils import get_rgb_color

    x_max = matrix_size
    y_max = matrix_size
    ply_data = open(("data/" + filename + ".ply"), "w")
    header = ply_header(x_max * y_max)
    ply_data.write(header)
    for x in range(0, x_max):
        for y in range(0, y_max):
            color = get_rgb_color(matrix[x][y])
            ply_data.write(str(x))
            ply_data.write(" ")
            ply_data.write(str(y))
            ply_data.write(" ")
            ply_data.write(str(matrix[x][y]))
            ply_data.write((" " + str(color[0]) + " " + str(color[1]) + " " + str(color[2])))
            ply_data.write("\n")
    ply_data.close()


def write_ply_file(matrix, x_max, y_max, filename):
    # saves a 2D numpy-array as PLY file.
    if not type(matrix) is np.ndarray: 
        raise TypeError("write_ply_file: matrix must be a numpy.ndarray!")
    if type(x_max) != int or type(y_max) != int: 
        raise TypeError("write_ply_file: x_max and y_max must be integers!") 
    if type(filename) != str: 
        raise TypeError("write_ply_file: filename must be a string!")

    from utils import get_rgb_color

    ply_data = open((filename + ".ply"), "w")
    header = ply_header(x_max * y_max)
    ply_data.write(header)
    for x in range(0, x_max):
        for y in range(0, y_max):
            color = get_rgb_color(matrix[x][y])
            ply_data.write(str(x))
            ply_data.write(" ")
            ply_data.write(str(y))
            ply_data.write(" ")
            ply_data.write(str(matrix[x][y]))
            ply_data.write((" " + str(color[0]) + " " + str(color[1]) + " " + str(color[2])))
            ply_data.write("\n")
    ply_data.close()




def xyz_to_ply(xyz_filename, ply_filename, colored=False, max_color=1021): 
    # opens an XYZ file and saves it as a PLY file.
    if type(xyz_filename) != str: 
        raise TypeError("xyz_to_ply: xyz_filename must be a string!")
    if type(ply_filename) != str: 
        raise TypeError("xyz_to_ply: ply_filename must be a string!")
    if type(colored) != bool: 
        raise TypeError("xyz_to_ply: colored must be boolean!")
    if type(max_color) != int: 
        raise TypeError("xyz_to_ply: max_color must be an integer!")

    from utils import get_rgb_color

    xyz_data = get_xyz_data_from_file(xyz_filename)
    if colored == True:
        z_max = xyz_data[0][2] 
        z_min = xyz_data[0][2]
        for i in xyz_data: 
            if i[2] > z_max: 
                z_max = i[2]
            if i[2] < z_min: 
                z_min = i[2] 
        color_list = []
        for i in xyz_data: 
            color = i[2] - z_min 
            color *= max_color/z_max 
            color = get_rgb_color(color)
            color_list.append(color)
    header = ply_header(len(xyz_data))
    header = header[:-1]

    with open(ply_filename, 'w') as ply_file: 
        ply_file.write(header) 

    for i in range(len(xyz_data)):     
        with open(ply_filename, 'a') as outfile: 
            if colored == False: 
                line = "\n{} {} {} {}".format(xyz_data[i][0], xyz_data[i][1], xyz_data[i][2], "0.0 0.0 0.0")
            else: 
                line = "\n{} {} {} {} {} {}".format(xyz_data[i][0], xyz_data[i][1], xyz_data[i][2], int(color_list[i][0]), int(color_list[i][1]), int(color_list[i][2]))
            outfile.write(line) 
            




def xyz_to_matrix(xyz_data, resolution):
    x_min = xyz_data[0][0]
    x_max = xyz_data[0][0]
    y_min = xyz_data[0][1]
    y_max = xyz_data[0][1]
    for x in range(len(xyz_data)):
        if x_min > xyz_data[x][0]:
            x_min = xyz_data[x][0]
        if x_max < xyz_data[x][0]:
            x_max = xyz_data[x][0]
        if y_min > xyz_data[x][1]:
            y_min = xyz_data[x][1]
        if y_max < xyz_data[x][1]:
            y_max = xyz_data[x][1]
    #print(x_min, " ", x_max, " ", y_min, " ", y_max)
    matrix_width = int((x_max - x_min)/resolution + 1.5) 
    matrix_heigth = int((y_max - y_min)/resolution + 1.5) 
    #print("matrix_width = ", matrix_width, "\tmatrix_heigth = ", matrix_heigth, "\n")
    new_matrix = np.zeros([matrix_width, matrix_heigth])
    for x in range(len(xyz_data)):
        i = int((1/resolution) * (xyz_data[x][0] - x_min))
        j = int((1/resolution) * (xyz_data[x][1] - y_min))
        #print("i = ", i, "\tj = ", j)
        new_matrix[i][j] = xyz_data[x][2]
        #print(new_matrix)
    return new_matrix


#Alle bounding-boxen aus allen trainingsmodellen in einem csv-File konkatiniert
def write_csv_file(matrix_size, csv_file_name, filename, bb_list, first=True, file_type='png'):
    #filename,width,height,class,xmin,ymin,xmax,ymax
    #                             [0] [2] [1] [3]
    n = len(bb_list)
    #y_max = matrix_size
    data = open(("data/" + csv_file_name + ".csv"), "a")
    if first == True:
        data.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
    if first == False:
        data.write("\n")
    for x in range(0, n):
        data.write((filename + "_mounds." + file_type))
        data.write(",")
        data.write(str(matrix_size))
        data.write(",")
        data.write(str(matrix_size))            
        data.write(",")
        data.write("gravemound")
        data.write(",")
        data.write(str(bb_list[x][0]))
        data.write(",")
        data.write(str(bb_list[x][2]))
        data.write(",")
        data.write(str(bb_list[x][1]))
        data.write(",")
        data.write(str(bb_list[x][3]))
        if x != n - 1:
            data.write("\n")
    data.close()




def write_png_file(matrix, filename, color=50.0, color_mode='RGB'):
    from utils import get_rgb_color
    x_max = len(matrix)
    y_max = len(matrix[0])
    #print([x_max, y_max])
    if color_mode == 'RGB':
        matrix_as_image = np.zeros((x_max, y_max, 3), dtype=np.uint8)
        for x in range(0, x_max):
            for j in range(0, y_max):
                #Rohwert um Faktor 50 erhöht!                
                matrix_as_image[x][j] = get_rgb_color(int(matrix[x][j] * color))
                
        img = Image.fromarray(matrix_as_image, color_mode)
        img.save(filename)        
    # TODO:
    elif color_mode == 'Greys':
        pass 
    else:
        pass 

def write_png_file_2(m, filename, max_color=1021):
    from utils import get_rgb_color
    matrix = m.copy()
    x_max = len(matrix)
    y_max = len(matrix[0])
    matrix -= np.min(matrix) 
    matrix *= max_color/np.max(matrix) #3021 2521    
    
    matrix_as_image = np.zeros((x_max, y_max, 3), dtype=np.uint8)
    for x in range(0, x_max):
        for j in range(0, y_max):            
            matrix_as_image[x][j] = get_rgb_color(int(matrix[x][j]))
            
    img = Image.fromarray(matrix_as_image, mode="RGB")
    img.save(filename)





def create_h5py_segmentation_dataset(filename): 
    from DTM_frame_data_test import get_DTM_frame_data 
    import h5py 

    data = get_DTM_frame_data()

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', data=x_train)
    seg_file.create_dataset('y_train', data=y_train)
    seg_file.create_dataset('x_test', data=x_test)
    seg_file.create_dataset('y_test', data=y_test)

    seg_file.close()


def create_compressed_h5py_segmentation_dataset(filename, bbox_manipulation=2, min_width=8, outlier_detection=False): 
    if outlier_detection == False:
        from DTM_frame_data_test import get_DTM_frame_data 
    else: 
        from DTM_frame_data_test_outliers import get_DTM_frame_data 
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)

    seg_file.close()


def create_compressed_h5py_segmentation_dataset_bigger(filename, bbox_manipulation=2, min_width=8, outlier_detection=False): 
    if outlier_detection == False:
        from DTM_frame_data_test import get_DTM_frame_data_bigger 
    else: 
        from DTM_frame_data_test_outliers import get_DTM_frame_data 
    import h5py 

    data = get_DTM_frame_data_bigger(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, frame_steps_norway=30)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)

    seg_file.close()




def create_compressed_h5py_pcr_dataset(filename, bbox_manipulation=2):     
    from DTM_pcr_data import get_DTM_pcr_data    
    import h5py 
    outlier_detection = False
    data = get_DTM_pcr_data(bbox_manipulation=bbox_manipulation, outlier_detection=outlier_detection)
    # [[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 
    x_train = []
    y_train_class = [] 
    y_train_reg = []
    for dtm in [dtm_germany_train, dtm_norway_train]: 
        x_train += dtm.X
        y_train_class += dtm.Y_c 
        y_train_reg += dtm.Y_r

    x_test = []
    y_test_class = [] 
    y_test_reg = []

    for dtm in [dtm_germany_test, dtm_norway_test]: 
        x_test += dtm.X 
        y_test_class += dtm.Y_c 
        y_test_reg += dtm.Y_r


    #print(len(x_train), len(x_train[0]), len(x_train[1]))
    print(len(x_train))
    for m in [x_train, x_test]: 
        for x in m: 
            if x.shape != (64, 64):
                print(x.shape)
    #print(x_train[1])
    for m in [x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg]: 
        print(len(m), m[0].shape)
    zero_count = 0
    for m in [y_test_class, y_train_class]: 
        for y in m: 
            if np.max(y) < 0.5: 
                zero_count += 1 
    print("Zero Count: ", zero_count)

    #x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    #y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    #x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    #y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))
    
    seg_file = h5py.File(filename, "w") 
    
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train_class', compression="gzip", data=y_train_class)
    seg_file.create_dataset('y_train_reg', compression="gzip", data=y_train_reg)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test_class', compression="gzip", data=y_test_class)
    seg_file.create_dataset('y_test_reg', compression="gzip", data=y_test_reg)
    seg_file.close()
    
    """
    seg_file.create_dataset('x_train', compression="gzip", data=np.asarray(x_train))
    seg_file.create_dataset('y_train_class', compression="gzip", data=np.asarray(y_train_class))
    seg_file.create_dataset('y_train_reg', compression="gzip", data=np.asarray(y_train_reg))
    seg_file.create_dataset('x_test', compression="gzip", data=np.asarray(x_test))
    seg_file.create_dataset('y_test_class', compression="gzip", data=np.asarray(y_test_class))
    seg_file.create_dataset('y_test_reg', compression="gzip", data=np.asarray(y_test_reg))
    """
    

def create_compressed_h5py_pcr_dataset_without_lower_saxony(filename, bbox_manipulation=2):     
    from DTM_detection_data_without_lower_saxony import get_DTM_pcr_data    
    import h5py 
    outlier_detection = False
    data = get_DTM_pcr_data(bbox_manipulation=bbox_manipulation, outlier_detection=outlier_detection)
    # [[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 
    x_train = []
    y_train_class = [] 
    y_train_reg = []
    for dtm in [dtm_germany_train, dtm_norway_train]: 
        x_train += dtm.X
        y_train_class += dtm.Y_c 
        y_train_reg += dtm.Y_r

    x_test = []
    y_test_class = [] 
    y_test_reg = []

    for dtm in [dtm_germany_test, dtm_norway_test]: 
        x_test += dtm.X 
        y_test_class += dtm.Y_c 
        y_test_reg += dtm.Y_r


    #print(len(x_train), len(x_train[0]), len(x_train[1]))
    print(len(x_train))
    for m in [x_train, x_test]: 
        for x in m: 
            if x.shape != (64, 64):
                print(x.shape)
    #print(x_train[1])
    for m in [x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg]: 
        print(len(m), m[0].shape)
    zero_count = 0
    for m in [y_test_class, y_train_class]: 
        for y in m: 
            if np.max(y) < 0.5: 
                zero_count += 1 
    print("Zero Count: ", zero_count)

    #x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    #y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    #x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    #y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))
    
    seg_file = h5py.File(filename, "w") 
    
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train_class', compression="gzip", data=y_train_class)
    seg_file.create_dataset('y_train_reg', compression="gzip", data=y_train_reg)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test_class', compression="gzip", data=y_test_class)
    seg_file.create_dataset('y_test_reg', compression="gzip", data=y_test_reg)
    seg_file.close()
    
    """
    seg_file.create_dataset('x_train', compression="gzip", data=np.asarray(x_train))
    seg_file.create_dataset('y_train_class', compression="gzip", data=np.asarray(y_train_class))
    seg_file.create_dataset('y_train_reg', compression="gzip", data=np.asarray(y_train_reg))
    seg_file.create_dataset('x_test', compression="gzip", data=np.asarray(x_test))
    seg_file.create_dataset('y_test_class', compression="gzip", data=np.asarray(y_test_class))
    seg_file.create_dataset('y_test_reg', compression="gzip", data=np.asarray(y_test_reg))
    """




def create_open_detection_dataset(filename, bbox_manipulation=2):     
    from DTM_open_detection_dataset import get_DTM_pcr_data    
    import h5py 
    outlier_detection = False
    data = get_DTM_pcr_data(bbox_manipulation=bbox_manipulation, outlier_detection=outlier_detection)
    # [[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 
    x_train = []
    y_train_class = [] 
    y_train_reg = []
    for dtm in [dtm_germany_train, dtm_norway_train]: 
        x_train += dtm.X
        y_train_class += dtm.Y_c 
        y_train_reg += dtm.Y_r

    x_test = []
    y_test_class = [] 
    y_test_reg = []

    for dtm in [dtm_germany_test, dtm_norway_test]: 
        x_test += dtm.X 
        y_test_class += dtm.Y_c 
        y_test_reg += dtm.Y_r


    #print(len(x_train), len(x_train[0]), len(x_train[1]))
    print(len(x_train))
    for m in [x_train, x_test]: 
        for x in m: 
            if x.shape != (64, 64):
                print(x.shape)
    #print(x_train[1])
    for m in [x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg]: 
        print(len(m), m[0].shape)
    zero_count = 0
    for m in [y_test_class, y_train_class]: 
        for y in m: 
            if np.max(y) < 0.5: 
                zero_count += 1 
    print("Zero Count: ", zero_count)

    #x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    #y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    #x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    #y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))
    
    seg_file = h5py.File(filename, "w") 
    
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train_class', compression="gzip", data=y_train_class)
    seg_file.create_dataset('y_train_reg', compression="gzip", data=y_train_reg)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test_class', compression="gzip", data=y_test_class)
    seg_file.create_dataset('y_test_reg', compression="gzip", data=y_test_reg)
    seg_file.close()
    
    """
    seg_file.create_dataset('x_train', compression="gzip", data=np.asarray(x_train))
    seg_file.create_dataset('y_train_class', compression="gzip", data=np.asarray(y_train_class))
    seg_file.create_dataset('y_train_reg', compression="gzip", data=np.asarray(y_train_reg))
    seg_file.create_dataset('x_test', compression="gzip", data=np.asarray(x_test))
    seg_file.create_dataset('y_test_class', compression="gzip", data=np.asarray(y_test_class))
    seg_file.create_dataset('y_test_reg', compression="gzip", data=np.asarray(y_test_reg))
    """





def create_open_detection_dataset_2(filename, bbox_manipulation=2):     
    from DTM_open_detection_dataset_2 import get_DTM_pcr_data    
    import h5py 
    outlier_detection = False
    data = get_DTM_pcr_data(bbox_manipulation=bbox_manipulation, outlier_detection=outlier_detection)
    # [[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 
    x_train = []
    y_train_class = [] 
    y_train_reg = []
    for dtm in [dtm_germany_train, dtm_norway_train]: 
        x_train += dtm.X
        y_train_class += dtm.Y_c 
        y_train_reg += dtm.Y_r

    x_test = []
    y_test_class = [] 
    y_test_reg = []

    for dtm in [dtm_germany_test, dtm_norway_test]: 
        x_test += dtm.X 
        y_test_class += dtm.Y_c 
        y_test_reg += dtm.Y_r


    #print(len(x_train), len(x_train[0]), len(x_train[1]))
    print(len(x_train))
    for m in [x_train, x_test]: 
        for x in m: 
            if x.shape != (64, 64):
                print(x.shape)
    #print(x_train[1])
    for m in [x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg]: 
        print(len(m), m[0].shape)
    zero_count = 0
    for m in [y_test_class, y_train_class]: 
        for y in m: 
            if np.max(y) < 0.5: 
                zero_count += 1 
    print("Zero Count: ", zero_count)    
    
    seg_file = h5py.File(filename, "w") 
    
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train_class', compression="gzip", data=y_train_class)
    seg_file.create_dataset('y_train_reg', compression="gzip", data=y_train_reg)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test_class', compression="gzip", data=y_test_class)
    seg_file.create_dataset('y_test_reg', compression="gzip", data=y_test_reg)
    seg_file.close()
    
    


def create_halv_open_detection_dataset(filename, bbox_manipulation=2):     
    from DTM_open_detection_dataset_3 import get_DTM_pcr_data    
    import h5py 
    outlier_detection = False
    data = get_DTM_pcr_data(bbox_manipulation=bbox_manipulation, outlier_detection=outlier_detection)
    # [[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]
    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 
    x_train = []
    y_train_class = [] 
    y_train_reg = []
    for dtm in [dtm_germany_train, dtm_norway_train]: 
        x_train += dtm.X
        y_train_class += dtm.Y_c 
        y_train_reg += dtm.Y_r

    x_test = []
    y_test_class = [] 
    y_test_reg = []

    for dtm in [dtm_germany_test, dtm_norway_test]: 
        x_test += dtm.X 
        y_test_class += dtm.Y_c 
        y_test_reg += dtm.Y_r


    #print(len(x_train), len(x_train[0]), len(x_train[1]))
    print(len(x_train))
    for m in [x_train, x_test]: 
        for x in m: 
            if x.shape != (64, 64):
                print(x.shape)
    #print(x_train[1])
    for m in [x_train, x_test, y_train_class, y_test_class, y_train_reg, y_test_reg]: 
        print(len(m), m[0].shape)
    zero_count = 0
    for m in [y_test_class, y_train_class]: 
        for y in m: 
            if np.max(y) < 0.5: 
                zero_count += 1 
    print("Zero Count: ", zero_count)    
    
    seg_file = h5py.File(filename, "w") 
    
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train_class', compression="gzip", data=y_train_class)
    seg_file.create_dataset('y_train_reg', compression="gzip", data=y_train_reg)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test_class', compression="gzip", data=y_test_class)
    seg_file.create_dataset('y_test_reg', compression="gzip", data=y_test_reg)
    seg_file.close()
    



def create_compressed_h5py_segmentation_dataset_2(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_frame_data_test_smaller_test_data import get_DTM_frame_data     
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()


def create_compressed_h5py_segmentation_dataset_without_lower_saxony(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_segmentation_data_without_lower_saxony import get_DTM_frame_data  
    
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()


def create_open_segmentation_dataset_small(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    #from DTM_open_segmentation_set import get_DTM_frame_data  # standard, with errors
    from DTM_open_segmentation_dataset_2 import get_DTM_frame_data  
    
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()



def create_halv_open_segmentation_dataset_small(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    #from DTM_open_segmentation_set import get_DTM_frame_data  # standard, with errors
    # train data with data thats not allowed for publication!
    from DTM_open_segmentation_dataset_3 import get_DTM_frame_data  
    
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()






def create_open_DTM1_segmentation_dataset_small(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_open_segmentation_set_DTM1 import get_DTM_frame_data  
    
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()




def create_compressed_h5py_segmentation_dataset_2_bigger(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_frame_data_test_smaller_test_data import get_DTM_frame_data_bigger     
    import h5py 

    data = get_DTM_frame_data_bigger(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()




def create_compressed_h5py_segmentation_dataset_no_lower_saxony(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_segmentation_data_without_lower_saxony import get_DTM_frame_data_bigger     
    import h5py 

    data = get_DTM_frame_data_bigger(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()


def create_open_segmentation_dataset_big(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    #from DTM_open_segmentation_set import get_DTM_frame_data_bigger  # standard, with errors
    from DTM_open_segmentation_dataset_2 import get_DTM_frame_data_bigger 
    import h5py 

    data = get_DTM_frame_data_bigger(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()



def create__halv_open_segmentation_dataset_big(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!    
    # train data with data thats not allowed for publication!
    from DTM_open_segmentation_dataset_3 import get_DTM_frame_data_bigger 
    import h5py 

    data = get_DTM_frame_data_bigger(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()




def create_mini_halv_open_segmentation_dataset(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):         
    # train data with data thats not allowed for publication!
    from DTM_mini_open_segmentation_data import get_DTM_frame_data 
    import h5py 

    data = get_DTM_frame_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()




def create_open_DTM1_segmentation_dataset_big(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_open_segmentation_set_DTM1 import get_DTM_frame_data_bigger     
    import h5py 

    data = get_DTM_frame_data_bigger(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()







def create_compressed_h5py_crxy_dataset(filename, bbox_manipulation=2, min_width=8, outlier_detection=False, x_shift=0, y_shift=0):     
    # more data in train- and less in test set!!!
    from DTM_crxy_data import get_DTM_crxy_data    
    import h5py 

    data = get_DTM_crxy_data(bbox_manipulation=bbox_manipulation, min_width=min_width, outlier_detection=outlier_detection, x_shift=x_shift, y_shift=y_shift)

    dtm_germany_train = data[0][0] 
    dtm_germany_test  = data[0][1] 
    dtm_norway_train  = data[1][0] 
    dtm_norway_test   = data[1][1] 

    x_train = np.concatenate((dtm_germany_train.X, dtm_norway_train.X)) 
    y_train = np.concatenate((dtm_germany_train.Y, dtm_norway_train.Y)) 
    x_test = np.concatenate((dtm_germany_test.X, dtm_norway_test.X))
    y_test = np.concatenate((dtm_germany_test.Y, dtm_norway_test.Y))

    seg_file = h5py.File(filename, "w") 
    seg_file.create_dataset('x_train', compression="gzip", data=x_train)
    seg_file.create_dataset('y_train', compression="gzip", data=y_train)
    seg_file.create_dataset('x_test', compression="gzip", data=x_test)
    seg_file.create_dataset('y_test', compression="gzip", data=y_test)
    print(len(x_train), len(x_test))
    seg_file.close()


# @njit
def read_h5py_segmentation_dataset(filename): 
    import h5py 
    seg_file = h5py.File(filename, "r") 
    x_train = seg_file.get("x_train")
    x_train = np.array(x_train)
    y_train = seg_file.get("y_train")
    y_train = np.array(y_train)
    x_test = seg_file.get("x_test") 
    x_test = np.array(x_test)
    y_test = seg_file.get("y_test") 
    y_test = np.array(y_test)
    seg_file.close() 
    print(x_train.shape[0], x_test.shape[0])
    return x_train, y_train, x_test, y_test


def read_h5py_pcr_dataset(filename): 
    import h5py 
    seg_file = h5py.File(filename, "r") 
    x_train = seg_file.get("x_train")
    x_train = np.array(x_train)
    y_train_class = seg_file.get("y_train_class")
    y_train_class = np.array(y_train_class)
    y_train_reg = seg_file.get("y_train_reg")
    y_train_reg = np.array(y_train_reg)
    x_test = seg_file.get("x_test") 
    x_test = np.array(x_test)
    y_test_class = seg_file.get("y_test_class") 
    y_test_class = np.array(y_test_class)
    y_test_reg = seg_file.get("y_test_reg") 
    y_test_reg = np.array(y_test_reg)
    seg_file.close() 
    print(len(x_train), len(x_test))
    return x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg 



def create_open_train_and_test_segmentation_sets(filename): 
    import h5py 
    x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset(filename) # for publication
    train = "S1_train.h5"
    test  = "S1_test.h5"
    #print(type(x_train))
    with h5py.File(train, "w") as seg_file: 
        seg_file.create_dataset('x_train', compression="gzip", compression_opts=9, data=x_train)
        print("1")
        seg_file.create_dataset('y_train', compression="gzip", compression_opts=9, data=y_train)
        print("2")
    with h5py.File(test, "w") as seg_file: 
        seg_file.create_dataset('x_test', compression="gzip", compression_opts=9, data=x_test)
        print("3")
        seg_file.create_dataset('y_test', compression="gzip", compression_opts=9, data=y_test)
        print("4")
        #print(len(x_train), len(x_test))
        #seg_file.close()
    
def read_segmentation_dataset(filename, train=False, test=False): 
    import h5py 
    
    with h5py.File(filename, "r") as seg_file:  
        if train == True and test == False:
            x_train = seg_file.get("x_train")
            x_train = np.array(x_train)
            y_train = seg_file.get("y_train")
            y_train = np.array(y_train)
            return x_train, y_train
        elif test == True and train == False:
            x_test = seg_file.get("x_test") 
            x_test = np.array(x_test)
            y_test = seg_file.get("y_test") 
            y_test = np.array(y_test)
            return x_test, y_test
        else: 
            x_train = seg_file.get("x_train")
            x_train = np.array(x_train)
            y_train = seg_file.get("y_train")
            y_train = np.array(y_train)
            x_test = seg_file.get("x_test") 
            x_test = np.array(x_test)
            y_test = seg_file.get("y_test") 
            y_test = np.array(y_test)
            return x_train, y_train, x_test, y_test
    
    


def create_open_train_and_test_detection_sets(filename): 
    import h5py 
    
    
    x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg = read_h5py_pcr_dataset(filename)
    train = "D1_train.h5"
    test  = "D1_test.h5"
    with h5py.File(train, "w") as seg_file:     
        seg_file.create_dataset('x_train', compression="gzip", compression_opts=9, data=x_train)
        print("1")
        seg_file.create_dataset('y_train_class', compression="gzip", compression_opts=9, data=y_train_class)
        print("2")
        seg_file.create_dataset('y_train_reg', compression="gzip", compression_opts=9, data=y_train_reg)
        print("3")
    with h5py.File(test, "w") as seg_file:  
        seg_file.create_dataset('x_test', compression="gzip", compression_opts=9, data=x_test)
        print("4")
        seg_file.create_dataset('y_test_class', compression="gzip", compression_opts=9, data=y_test_class)
        print("5")
        seg_file.create_dataset('y_test_reg', compression="gzip", compression_opts=9, data=y_test_reg)
        print("6")
    



def read_detection_dataset(filename, train=False, test=False):
    import h5py 

    with h5py.File(filename, "r") as seg_file:  
        if train == True and test == False:
            x_train = seg_file.get("x_train")
            x_train = np.array(x_train)
            y_train_class = seg_file.get("y_train_class")
            y_train_class = np.array(y_train_class)
            y_train_reg = seg_file.get("y_train_reg")
            y_train_reg = np.array(y_train_reg)
            return x_train, y_train_class, y_train_reg

        elif test == True and train == False:
            x_test = seg_file.get("x_test") 
            x_test = np.array(x_test)
            y_test_class = seg_file.get("y_test_class") 
            y_test_class = np.array(y_test_class)
            y_test_reg = seg_file.get("y_test_reg") 
            y_test_reg = np.array(y_test_reg)
            return x_test, y_test_class, y_test_reg 

        else:
            x_train = seg_file.get("x_train")
            x_train = np.array(x_train)
            y_train_class = seg_file.get("y_train_class")
            y_train_class = np.array(y_train_class)
            y_train_reg = seg_file.get("y_train_reg")
            y_train_reg = np.array(y_train_reg)
            x_test = seg_file.get("x_test") 
            x_test = np.array(x_test)
            y_test_class = seg_file.get("y_test_class") 
            y_test_class = np.array(y_test_class)
            y_test_reg = seg_file.get("y_test_reg") 
            y_test_reg = np.array(y_test_reg)
            return x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg 
    
    








if __name__ == "__main__": 
    from preprocessing import transform
    #create_compressed_h5py_segmentation_dataset("frame_segmentation_dataset_outlier_detected_moundness_st.h5", 
    #                                            bbox_manipulation=0, 
    #                                            outlier_detection=True  )
    """
    create_compressed_h5py_segmentation_dataset_2("frame_segmentation_dataset_smaller_test_set_no_min_radius.h5", 
                                                min_width=0,
                                                bbox_manipulation=0, 
                                                outlier_detection=False) # train: 15872 test: 7256
    """


    """
    index = 1
    for i in range(1, 31, 3): 
        for j in range(1, 31, 3):
            create_compressed_h5py_segmentation_dataset_2("frame_segmentation_dataset_smaller_test_set_no_min_radius_" + str(index) + ".h5", 
                                                        min_width=0,
                                                        bbox_manipulation=0, 
                                                        outlier_detection=False, 
                                                        x_shift=i, 
                                                        y_shift=j ) 
            index += 1
    """
    #create_compressed_h5py_crxy_dataset("crxy_dataset_smaller_test_set_no_min_radius.h5", 
    #                                    min_width=0,
    #                                    bbox_manipulation=0, 
    #                                    outlier_detection=False)


    # for detection: 
    #create_compressed_h5py_pcr_dataset("pcr_dataset_smaller_test_set.h5", bbox_manipulation=0)

    #create_compressed_h5py_segmentation_dataset_2_bigger("frame_segmentation_dataset_smaller_test_set_no_min_radius_bigger.h5", 
    #                                            min_width=0,
    #                                            bbox_manipulation=0, 
    #                                            outlier_detection=False) # train: 59680 test: 29024

    #create_compressed_h5py_segmentation_dataset_2_bigger("frame_segmentation_dataset_smaller_test_set_no_min_radius_bigger2.h5", 
    #                                            min_width=0,
    #                                            bbox_manipulation=0, 
    #                                            outlier_detection=False) # train: 59680 test: 29024
    
    
    """
    create_compressed_h5py_segmentation_dataset_without_lower_saxony("frame_segmentation_dataset_smaller_test_set_no_lower_saxony.h5", 
                                                min_width=0,
                                                bbox_manipulation=0, 
                                                outlier_detection=False) # train: 14592, test: 5320

    create_compressed_h5py_segmentation_dataset_no_lower_saxony("frame_segmentation_dataset_without_lower_saxony.h5", 
                                                min_width=0,
                                                bbox_manipulation=0, 
                                                outlier_detection=False) # train: 54720, test: 21280


    create_compressed_h5py_pcr_dataset_without_lower_saxony("pcr_dataset_without_lower_saxony.h5", bbox_manipulation=0) # train: 34560, test: 19216
    """
    #read_h5py_segmentation_dataset("frame_segmentation_dataset_without_lower_saxony.h5")
    #read_h5py_pcr_dataset("pcr_dataset_without_lower_saxony.h5") 
    #read_h5py_pcr_dataset("pcr_dataset_smaller_test_set.h5") # train: 35584, test: 21152
    #x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg = read_h5py_pcr_dataset("pcr_dataset_smaller_test_set.h5")
    #print(len(x_train), len(x_test))
    """
    create_open_segmentation_dataset_small("OpenSegmentationDatasetSmall.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False ) # 11328 8584

    create_open_segmentation_dataset_big(   "OpenSegmentationDatasetBig.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False ) # 43264 32736
    """


    """
    create_open_DTM1_segmentation_dataset_small("OpenSegmentationDatasetSmallDTM1.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False ) # 13159 10320

    create_open_DTM1_segmentation_dataset_big(   "OpenSegmentationDatasetBigDTM1.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False ) # 52220 40431
    """
    #create_open_detection_dataset("OpenDetectionDataset.h5", bbox_manipulation=0)
    """
    create_open_segmentation_dataset_small("openSegmentationDataset2_small.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False )  # 18848 8584

    create_open_segmentation_dataset_big("openSegmentationDataset2_big.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False ) # 74624 32736
    """
    #create_open_detection_dataset_2("OpenDetectionDataset_2.h5", bbox_manipulation=0)

    """
    create__halv_open_segmentation_dataset_big("halv_openSegmentationDataset_big.h5", 
                                            min_width=0,
                                            bbox_manipulation=0, 
                                            outlier_detection=False ) # 87328 32736

    """
    #create_halv_open_detection_dataset("HalvOpenDetectionDataset.h5", bbox_manipulation=0) # 32272 29584
    #filename = "openSegmentationDataset2_big.h5" # open
    #filename = "OpenDetectionDataset_2.h5"
    #filename = "HalvOpenDetectionDataset.h5" # halv open
    #x_train, y_train_class, y_train_reg, x_test, y_test_class, y_test_reg = read_h5py_pcr_dataset(filename)
    """
    for i in range(5): 
        plt.subplot(1,2,1)
        plt.imshow(transform(x_test[-(200*i)], "laplacian")) 
        plt.subplot(1,2,2) 
        plt.imshow(y_test_class[-(200*i)])
        plt.show()
    """

    #create_mini_halv_open_segmentation_dataset("MiniDataset_halv_open.h5", 
    #                                        min_width=0,
    #                                        bbox_manipulation=0, 
    #                                        outlier_detection=False ) # 565 111

    #create_open_train_and_test_segmentation_sets()
    #create_open_train_and_test_detection_sets()
    """
    x_train, y_train = read_segmentation_dataset("S1_test.h5", train=False, test=True)
    print(x_train.shape)
    plt.imshow(x_train[0]) 
    plt.show()
    """
    x_train, y_train_class, y_train_reg = read_detection_dataset("D1_train.h5", train=True, test=False)
    print(x_train.shape)
    for i in range(100): 
        plt.imshow(x_train[i * 10]) 
        plt.show()

    """
    splitting in Terminal:
    split -b 12M "D1_train.h5" "D1_train.h5."
    split -b 12M "D1_test.h5" "D1_test.h5."
    split -b 12M "S1_train.h5" "S1_train.h5."
    split -b 12M "S1_test.h5" "S1_test.h5."
    """