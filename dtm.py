import numpy as np 
import matplotlib.pyplot as plt 
import file_ops, terrain
from bounding_box import BBox
from augment_by_smallsizing import smallsize_matrix_general
import sys
#import utils


class Coordinates:
    # geographical coordinates of a DTM. Do not confuse with the  
    # bounding-box coordinates within a matrix! 

    def __init__(self, min_north, min_east, max_north, max_east): 
        self.min_north = min_north 
        self.min_east = min_east 
        self.max_north = max_north 
        self.max_east = max_east
        # shape[0] refers to east, shape[1] refers to north!
        self.shape = [self.max_east - self.min_east, self.max_north - self.min_north]



class DTM: 
    def __init__(self, path, filename, file_format, resolution, xml_folder="", voc=False, target_resolution=None, coordinates=None): 
        self.path = path 
        self.filename = filename 
        self.format = file_format 
        self.resolution = resolution
        self.xml_folder = xml_folder
        self.is_voc = voc
        if target_resolution == None:
            self.target_resolution = self.resolution
        else: 
            self.target_resolution = target_resolution
        if coordinates: 
            self.coordinates = coordinates 
        else: 
            self.coordinates = None
        self.predicted_matrix = None


    def get_bboxes(self): 
        from file_ops import xml_to_bbox
        if self.xml_folder != "":
            self.bbox_list = xml_to_bbox(self.xml_folder, voc=self.is_voc)
        else: self.bbox_list = []
    

    def set_target_resolution(target_resolution): 
        self.target_resolution = target_resolution


    def set_matrix(self, resolution): 
        from augment_by_smallsizing import smallsize_matrix_general
        filename = self.path + "/" + self.filename + "." + self.format
        if self.format == "npy": 
            matrix = np.load(filename) 
            #matrix = smallsize_matrix_general(matrix, resolution) # Might cause issues! 
        elif self.format == "xyz": 
            xyz_data = file_ops.get_xyz_data_from_file(filename) 
            matrix = file_ops.xyz_to_matrix(xyz_data, self.resolution)
        elif self.format == "tif":
            matrix = file_ops.geotif_to_matrix(filename)
        else: 
            print("No valid format: {}".format(self.format)) 
            sys.exit()
        self.matrix = matrix


    def get_absolute_position(self, i, j): 
        # returns the geographical coordinates of element (i, j) in the DTM matrix. 
        # TODO: The result deviates about -1 for c_north and +1 for c_east! 
        if not self.coordinates: 
            print("Warning: DTM doesn't have geographical coordinates!") 
            return None 
        else: 
            min_north = self.coordinates.min_north 
            max_north = self.coordinates.max_north 
            min_east  = self.coordinates.min_east 
            max_east  = self.coordinates.max_east
            m_shape = self.matrix.shape 
            c_shape = self.coordinates.shape
            
            c_north = round(min_north + i * (c_shape[0] / m_shape[0]))
            c_east  = round( min_east + j * (c_shape[1] / m_shape[1])) 
            return c_north, c_east 
             

    def get_window_list_by_position(self, resolution, non_gravemounds=False):
        window_list = []  
        if not self.matrix: 
            matrix = self.set_matrix(resolution)   
        if not self.bbox_list: 
            self.bbox_list = self.get_bboxes()  
        for box in self.bbox_list:
            if box.label != "non_gravemound" or non_gravemounds == True:
                x_size = box.width 
                y_size = box.height
                box.print()
                window = get_window_by_position(y_size, x_size, matrix, box.ymin, box.xmin)[0]
                window -= np.min(window)               
                window_list.append(window)     
        self.window_list = window_list


    def get_distance(self, P1, P2): 
        #P1 and P2 are tuples of coordinates.
        a = (P1[0] - P2[0])**2 
        b = (P1[1] - P2[1])**2 
        return np.sqrt(a + b)  


    def outlier_detection(self, resolution, non_gravemounds=False): 
        from preprocessing import outlier_detection_oc_svm
        from utils import get_window_by_position
        window_list = []  
        #if not self.matrix: 
        #matrix = self.set_matrix(resolution) 
        matrix = self.matrix        
        print(matrix.shape)  
        if not self.bbox_list: 
            self.bbox_list = self.get_bboxes()  
        #try:
        #print(self.bbox_list)
        if type(self.bbox_list) != list: 
            self.bbox_list = []  
        for box in self.bbox_list:
            if box.label != "non_gravemound" or non_gravemounds == True:
                x_size = box.width 
                y_size = box.height
                #box.print()
                #print(matrix.shape)
                window = get_window_by_position(y_size, x_size, matrix, box.ymin, box.xmin)[0]
                window -= np.min(window)               
                window_list.append(window)  
    
        bbox_list = self.bbox_list
        if len(bbox_list) > 0:  
            print("BBox list before outlier detection: ", len(bbox_list))
            #try: 
            window_list, bbox_list = outlier_detection_oc_svm(window_list, bbox_list, mean_type="moundness", classification_type="st")  
            print("BBox list after outlier detection: ", len(bbox_list))
        #except: 
        #    raise Exception("Error in outlier_detection")
        #except: 
        #    bbox_list = []
        self.window_list = window_list
        self.bbox_list = bbox_list


    def set_label_matrix(self, bbox_manipulation=2, non_gravemounds=False, min_width=8): 
        bbox_list = self.bbox_list
        width = self.matrix.shape[0]
        height = self.matrix.shape[1]
        matrix = np.zeros((width, height))
        if bbox_manipulation != 0: 
            for i in range(len(bbox_list)): 
                bbox_list[i].manipulate_bbox(width, height, bbox_manipulation)
        #ymin, ymax, xmin, xmax
        # 0     1     2     3
        for bbox in bbox_list: 
            #if min(bb_width, bb_height) > 8 :
            if bbox.label != "non_gravemound" or non_gravemounds == True:             
                bb_width = bbox.width
                bb_height = bbox.height
                #if min(bb_width, bb_height) > 4 and max(bb_width, bb_height) < 27: 
                if min(bb_width, bb_height) >= min_width:
                    m = [bbox.ymin + (bb_width - 1) / 2, bbox.xmin + (bb_height - 1) / 2] 
                    r = max((bb_width - 1) / 2, (bb_height - 1) / 2)             
                    for i in range(bbox.ymin, bbox.ymax + 1): 
                        for j in range(bbox.xmin, bbox.xmax + 1): 
                            try:                       
                                d = self.get_distance(m, [i, j])
                                #print(m, [i, j], r, d)
                                if d <= r: 
                                    matrix[i][j] = 1
                            except: 
                                pass
        self.label_matrix = matrix

    def set_crxy_label_matrix(self, bbox_manipulation=2, non_gravemounds=False, min_width=8): 
        # Depricated! Do not use!
        # CRXY: Class, Radius, X-middle and Y-middle of grave mound
        bbox_list = self.bbox_list
        width = self.matrix.shape[0]
        height = self.matrix.shape[1]
        c_matrix = np.zeros((width, height))
        r_matrix = np.zeros((width, height))
        x_matrix = np.zeros((width, height)) 
        y_matrix = np.zeros((width, height))
        if bbox_manipulation != 0: 
            for i in range(len(bbox_list)): 
                bbox_list[i].manipulate_bbox(width, height, bbox_manipulation)
        #ymin, ymax, xmin, xmax
        # 0     1     2     3
        for bbox in bbox_list: 
            #if min(bb_width, bb_height) > 8 :
            if bbox.label != "non_gravemound" or non_gravemounds == True:             
                bb_width = bbox.width
                bb_height = bbox.height
                #if min(bb_width, bb_height) > 4 and max(bb_width, bb_height) < 27: 
                if min(bb_width, bb_height) >= min_width:
                    m = [bbox.ymin + (bb_width - 1) / 2, bbox.xmin + (bb_height - 1) / 2] 
                    r = max((bb_width - 1) / 2, (bb_height - 1) / 2) 
                    for i in range(bbox.ymin, bbox.ymax + 1): 
                        for j in range(bbox.xmin, bbox.xmax + 1): 
                            try:                       
                                d = self.get_distance(m, [i, j])
                                #print(m, [i, j], r, d)
                                if d <= r: 
                                    c_matrix[i][j] = 1
                                    r_matrix[i][j] = r
                                    x_matrix[i][j] = m[0] - i 
                                    y_matrix[i][j] = m[1] - j 
                            except: 
                                pass
        self.crxy_label_matrix = np.array([c_matrix, r_matrix, x_matrix, y_matrix])

    



    def show_segmentation_data(self):
        plt.subplot(2,1,1)
        plt.imshow(self.matrix) 
        plt.subplot(2,1,2) 
        plt.imshow(self.label_matrix)
        plt.show()

    
    def get_window_by_position(self, x_size, y_size, x_start, y_start):
        # copies a rectangular shaped area out of the matrix and returns it together with 
        # its bounding box.    
        matrix = self.matrix
        x_end = x_start + x_size
        y_end = y_start + y_size        
        bbox = BBox(x_start, x_end, y_start, y_end, "", None, -1, is_voc=self.is_voc)
        window = matrix[x_start:x_end, y_start:y_end]
        return [window, bbox]


    def create_frame_segmentation_data(self, kernel_size=16, augmentation=False, augmentation_steps=-1, frame_steps=7, x_shift=0, y_shift=0): 
        from utils import get_window_by_position
        #from preprocessing import matrix_derivative_2d_one_dir
        
        if augmentation == True:
            smallsized_matrices = [] 
            smallsized_label_matrices = [] 
            
            m = smallsize_matrix_general(self.matrix, augmentation_steps) 
            l = smallsize_matrix_general(self.label_matrix, augmentation_steps) 
            #print("m: ", len(m))
            smallsized_matrices = m 
            smallsized_label_matrices = l 
            #print("Length Augmentation Matrices: ", len(smallsized_matrices))
            matrices = smallsized_matrices 
            label_matrices = smallsized_label_matrices
        else: 
            matrices = [self.matrix]
            label_matrices = [self.label_matrix]
        
        #sys.exit()
        X = [] 
        Y = []    
        #for h in range(len(matrices)):
        for k in range(len(matrices)): 
            """
            plt.subplot(1,2,1)
            plt.imshow(matrices[k], cmap="Greys") 
            plt.subplot(1,2,2) 
            plt.imshow(label_matrices[k], cmap="Greys")
            plt.show()
            """
            #plt.imshow(matrices[k])
            #plt.show()
            matrix = matrices[k]
            #self.show_segmentation_data()
            #print(type(matrix), len(matrix))
            label_matrix = label_matrices[k]            
            for i in range(x_shift, matrix.shape[0] - kernel_size + 1, frame_steps): 
                for j in range(y_shift, matrix.shape[1] - kernel_size + 1, frame_steps):                 
                    matrix_window_and_box = get_window_by_position(kernel_size, kernel_size, matrix, i, j) 
                    mlabel_window_and_box = get_window_by_position(kernel_size, kernel_size, label_matrix, i, j)
                    m_window = matrix_window_and_box[0] 
                    l_window = mlabel_window_and_box[0]
                    m_window -= np.min(m_window)
                    l_window -= np.min(l_window)
                    X.append(m_window) 
                    Y.append(l_window)
            #print(k + 1, "von ", len(matrices), ":  ", len(X), len(Y))        
        print("Matrix: ", self.matrix.shape , " X: ", len(X), ", Y: ", len(Y))
        #if len(X) == 64: 
        #    plt.imshow(X[0]) 
        #    plt.show()
        
        X = np.asarray(X) 
        Y = np.asarray(Y)

        self.frame_segmentation_data = [X, Y]


    def create_mini_frame_segmentation_data(self, kernel_size=16, augmentation=False, augmentation_steps=-1, frame_steps=7, x_shift=0, y_shift=0): 
        from utils import get_window_by_position
        #from preprocessing import matrix_derivative_2d_one_dir
        
        if augmentation == True:
            smallsized_matrices = [] 
            smallsized_label_matrices = [] 
            
            m = smallsize_matrix_general(self.matrix, augmentation_steps) 
            l = smallsize_matrix_general(self.label_matrix, augmentation_steps) 
            #print("m: ", len(m))
            smallsized_matrices = [m[0]] 
            smallsized_label_matrices = [l[0]] 
            #print("Length Augmentation Matrices: ", len(smallsized_matrices))
            matrices = smallsized_matrices 
            label_matrices = smallsized_label_matrices
        else: 
            matrices = [self.matrix]
            label_matrices = [self.label_matrix]
        
        #sys.exit()
        X = [] 
        Y = []    
        #for h in range(len(matrices)):
        for k in range(len(matrices)): 
            """
            plt.subplot(1,2,1)
            plt.imshow(matrices[k], cmap="Greys") 
            plt.subplot(1,2,2) 
            plt.imshow(label_matrices[k], cmap="Greys")
            plt.show()
            """
            #plt.imshow(matrices[k])
            #plt.show()
            matrix = matrices[k]
            #self.show_segmentation_data()
            #print(type(matrix), len(matrix))
            label_matrix = label_matrices[k]            
            for i in range(x_shift, matrix.shape[0] - kernel_size + 1, frame_steps): 
                for j in range(y_shift, matrix.shape[1] - kernel_size + 1, frame_steps):                 
                    matrix_window_and_box = get_window_by_position(kernel_size, kernel_size, matrix, i, j) 
                    mlabel_window_and_box = get_window_by_position(kernel_size, kernel_size, label_matrix, i, j)
                    m_window = matrix_window_and_box[0] 
                    l_window = mlabel_window_and_box[0]
                    m_window -= np.min(m_window)
                    l_window -= np.min(l_window)
                    X.append(m_window) 
                    Y.append(l_window)
            #print(k + 1, "von ", len(matrices), ":  ", len(X), len(Y))        
        print("Matrix: ", self.matrix.shape , " X: ", len(X), ", Y: ", len(Y))
        #if len(X) == 64: 
        #    plt.imshow(X[0]) 
        #    plt.show()
        
        X = np.asarray(X) 
        Y = np.asarray(Y)

        self.frame_segmentation_data = [X, Y]





    def create_crxy_data(self, kernel_size=16, augmentation=False, augmentation_steps=-1, frame_steps=7, x_shift=0, y_shift=0): 
        # CRXY: Class, Radius, X-middle and Y-middle of grave mound
        from utils import get_window_by_position, get_crxy_window_by_position
        #from preprocessing import matrix_derivative_2d_one_dir
        
        if augmentation == True:
            smallsized_matrices = [] 
            smallsized_label_matrices = [] 
            
            m = smallsize_matrix_general(self.matrix, augmentation_steps) 
            l_list = [] 
            for c in self.crxy_label_matrix: 
                print(c.shape)
                l = smallsize_matrix_general(c, augmentation_steps)
                for l2 in l: 
                    l_list.append(l2)
                l_list.append(l) 
            for l in l_list: 
                print(l.shape)           
            smallsized_label_matrices = np.asarray(l_list)
            smallsized_matrices = m             
            matrices = smallsized_matrices 
            label_matrices = smallsized_label_matrices
        else: 
            matrices = [self.matrix]
            label_matrices = [self.crxy_label_matrix]        
        
        X = [] 
        Y = []            
        for k in range(len(matrices)):                 
            matrix = matrices[k]
            label_matrix = label_matrices[k]            
            for i in range(x_shift, matrix.shape[0] - kernel_size + 1, frame_steps): 
                for j in range(y_shift, matrix.shape[1] - kernel_size + 1, frame_steps):                 
                    matrix_window_and_box = get_window_by_position(kernel_size, kernel_size, matrix, i, j) 
                    mlabel_window_and_box = get_crxy_window_by_position(kernel_size, kernel_size, label_matrix, i, j)
                    m_window = matrix_window_and_box[0] 
                    l_window = mlabel_window_and_box[0]
                    m_window -= np.min(m_window)
                    l_window -= np.min(l_window)
                    X.append(m_window) 
                    Y.append(l_window)            
        
        X = np.asarray(X) 
        Y = np.asarray(Y)

        self.crxy_data = [X, Y]



    def create_classification_data(self, augmentation=False, augmentation_steps=-1): 
        # creates unaugmented and unpreprocessed labeled data as lists. 
        # the returned images have different sizes and have to be 
        # adjusted to fit as input into the classification network. 
        # The labels are strings and need one-hot-encoding. 
        from utils import get_window_by_position   

        bbox_list = self.bbox_list
        matrix = self.matrix           
        X = [] 
        Y = []    
        sizes = []   
              
        for bbox in bbox_list: 
            #print(bbox.width, bbox.height, bbox.label)                                
            window_and_box = get_window_by_position(bbox.height, bbox.width, matrix, bbox.ymin, bbox.xmin)  
            size = int((bbox.width + bbox.height) / 2.) 

            window = window_and_box[0] 
            #print(window) 
            
            #print(np.max(window) - np.min(window))    
            #except: 
            #    print(type(window))       
            #window -= np.min(window)
            if augmentation == False: 
                X.append(window)             
                Y.append(bbox.label)                     
                sizes.append(size) 
            else: 
                s_m = smallsize_matrix_general(window, augmentation_steps)
                s_l = [bbox.label] * len(s_m)                 
                for i in range(len(s_m)): 
                    X.append(s_m[i]) 
                    Y.append(s_l[i]) 
                    sizes.append(int((s_m[i].shape[0] + s_m[i].shape[1]) / 2.))
        #plt.imshow(matrix) 
        #plt.show()     
        sizes = np.asarray(sizes)
        self.classification_data = [X, Y, sizes]

    

    """
    def create_detection_data(self, csv_name, augmentation=False, augmentation_steps=-1, frame_size=64, frame_steps=12, filepath="", file_type="npy"): 
        from utils import get_window_by_position, get_frame_with_bbox_info
        #from preprocessing import matrix_derivative_2d_one_dir
        bbox_list = self.bbox_list
        matrix = self.matrix 
        index = 0
        csv_info = ""
        for i in range(0, matrix.shape[0] - frame_size + 1, frame_steps): 
            for j in range(0, matrix.shape[1] - frame_size + 1, frame_steps): 
                index += 1
                frame_ID = str(index)
                #print(index)
                # matrix, bbox_list, frame_xmin, frame_ymin, frame_ID, filepath="", kernel_size=64, file_type="npy"
                #print(matrix.shape)
                csv_info += get_frame_with_bbox_info(matrix, bbox_list, i, j, frame_ID=frame_ID, filepath=filepath, kernel_size=frame_size, file_type=file_type)
        csv_info = csv_info[:-1] 
        csv_file = open(csv_name, "w") 
        csv_file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
        csv_file.write(csv_info)
        csv_file.close()
        
    """


    def create_detection_data(self, csv_name, augmentation=False, augmentation_steps=-1, frame_size=64, frame_steps=12, filepath="", file_type="npy"): 
        from utils import get_window_by_position, get_frame_with_bbox_info
        #from preprocessing import matrix_derivative_2d_one_dir
        #bbox_list = self.bbox_list
        matrix = self.matrix 
        bbox_list = self.bbox_list.copy()
        bb_list = bbox_list.copy()

        #for b in range(len(bbox_list)):             
        #    print("BBox: ", bbox_list[b].xmin, bbox_list[b].ymin, bbox_list[b].xmax, bbox_list[b].ymax)
        
        index = 0
        csv_info = ""
        if augmentation: 
            new_frame_size = frame_size * augmentation_steps  
            for b in range(len(bb_list)): 
                #print("BBox b: ", bbox_list[b].xmin, bbox_list[b].ymin, bbox_list[b].xmax, bbox_list[b].ymax)
                bb_list[b].xmin = int(bb_list[b].xmin / augmentation_steps)
                bb_list[b].ymin = int(bb_list[b].ymin / augmentation_steps)
                bb_list[b].xmax = int(bb_list[b].xmax / augmentation_steps)
                bb_list[b].ymax = int(bb_list[b].ymax / augmentation_steps)
                bb_list[b].width = bb_list[b].xmax - bb_list[b].xmin
                bb_list[b].height = bb_list[b].ymax - bb_list[b].ymin

        else: 
            new_frame_size = frame_size

        for i in range(0, matrix.shape[0] - new_frame_size + 1, frame_steps): 
            for j in range(0, matrix.shape[1] - new_frame_size + 1, frame_steps): 
                #bbox_list = self.bbox_list.copy()
                #for b in range(len(bbox_list)): 
                #    print("BBox: ", bbox_list[b].xmin, bbox_list[b].ymin, bbox_list[b].xmax, bbox_list[b].ymax)
                index += 1
                frame_ID = str(index)
                #print(index)
                # matrix, bbox_list, frame_xmin, frame_ymin, frame_ID, filepath="", kernel_size=64, file_type="npy"
                #print(matrix.shape)
                csv_info += get_frame_with_bbox_info(   matrix, 
                                                        bb_list, 
                                                        i, 
                                                        j, 
                                                        frame_ID=frame_ID, 
                                                        filepath=filepath, 
                                                        kernel_size=new_frame_size, 
                                                        file_type=file_type, 
                                                        augmentation=augmentation, 
                                                        augmentation_steps=augmentation_steps)
        csv_info = csv_info[:-1] 
        csv_file = open(csv_name, "w") 
        csv_file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
        csv_file.write(csv_info)
        csv_file.close()       


    def create_fullsize_detection_data(self, csv_name, augmentation=False, augmentation_steps=-1, filepath="", file_type="npy"): 
        # creates detection data of individual size (the size of the original DTM matrix)
        from utils import get_window_by_position, get_matrix_with_bbox_info
        matrix = self.matrix 
        bbox_list = self.bbox_list.copy()
        bb_list = bbox_list.copy()
        
        index = 0
        csv_info = ""
        if augmentation:
            for b in range(len(bb_list)): 
                bb_list[b].xmin = int(bb_list[b].xmin / augmentation_steps)
                bb_list[b].ymin = int(bb_list[b].ymin / augmentation_steps)
                bb_list[b].xmax = int(bb_list[b].xmax / augmentation_steps)
                bb_list[b].ymax = int(bb_list[b].ymax / augmentation_steps)
                bb_list[b].width = bb_list[b].xmax - bb_list[b].xmin
                bb_list[b].height = bb_list[b].ymax - bb_list[b].ymin

        csv_info += get_matrix_with_bbox_info(  matrix, 
                                                bb_list,
                                                filepath=filepath,                                                 
                                                file_type=file_type, 
                                                augmentation=augmentation, 
                                                augmentation_steps=augmentation_steps)
        csv_info = csv_info[:-1] 
        csv_file = open(csv_name, "w") 
        csv_file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
        csv_file.write(csv_info)
        csv_file.close()

    
    def create_pcr_data(self, augmentation=False, augmentation_steps=-1, frame_size=64, frame_steps=12): 
        # pcr: pixelwise classification and regression
        from utils import get_pcr_frame_and_label 
        from augment_by_smallsizing import smallsize_matrix_general 
        
        matrix = self.matrix 
        bbox_list = self.bbox_list.copy()
        bb_list = bbox_list.copy()
        
        index = 0
        #pcr_data = []
        X = [] 
        Y_c = [] 
        Y_r = []
        if augmentation: 
            new_frame_size = frame_size * augmentation_steps 
            #if augmentation_steps == 2: 
            #    test_matrix = smallsize_matrix_general(matrix, augmentation_steps)[0]
            #    test_matrix2 = smallsize_matrix_general(matrix, augmentation_steps)[0]
            for b in range(len(bb_list)): 
                bb_list[b].xmin = int(bb_list[b].xmin / augmentation_steps)
                bb_list[b].ymin = int(bb_list[b].ymin / augmentation_steps)
                bb_list[b].xmax = int(bb_list[b].xmax / augmentation_steps)
                bb_list[b].ymax = int(bb_list[b].ymax / augmentation_steps)
                bb_list[b].width = bb_list[b].xmax - bb_list[b].xmin
                bb_list[b].height = bb_list[b].ymax - bb_list[b].ymin
                #if augmentation_steps == 2:   
                #    test_matrix2[bb_list[b].ymin:bb_list[b].ymax, bb_list[b].xmin:bb_list[b].xmax] += 10 
            """    
            if augmentation_steps == 2:
                plt.subplot(211) 
                plt.imshow(test_matrix) 
                plt.subplot(212) 
                plt.imshow(test_matrix2) 
                plt.show()
            """

        else: 
            new_frame_size = frame_size

        for i in range(0, matrix.shape[0] - new_frame_size + 1, frame_steps): 
            for j in range(0, matrix.shape[1] - new_frame_size + 1, frame_steps):                             
                
                input_frames, output_classes, output_radii = get_pcr_frame_and_label(matrix, bbox_list, i, j, augmentation=augmentation, augmentation_steps=augmentation_steps, kernel_size=new_frame_size) 
                if not [] in [input_frames + output_classes + output_radii]:
                    #pcr_data.append(frame_data)
                    X += input_frames 
                    Y_c += output_classes 
                    Y_r += output_radii
                    #pcr_data += frame_data
        """
        for f in pcr_data: 
            X.append(f[0])  
            Y_c.append(f[1])  
            Y_r.append(f[2]) 
        """
        self.pcr_data = [X, Y_c, Y_r] 
          









    
class DTMList: 
    def __init__(self, dtm_list, label_dict={}): 
        # dtm_list argument as simple python-list.
        # label_dict: contains label strings as keys pointing to an integer respectively. 
        self.dtm_list = dtm_list
    

    def get_classification_dataset(self, augmentation=True, augmentation_steps=2): 
        for dtm in self.dtm_list:
            dtm.set_matrix(dtm.resolution)
            dtm.get_bboxes()            
            dtm.create_classification_data(augmentation=augmentation, augmentation_steps=augmentation_steps)
        self.X = []
        self.Y = []
        self.sizes = []
        for dtm in self.dtm_list:
            for x in dtm.classification_data[0]: 
                self.X.append(x)
            for y in dtm.classification_data[1]: 
                self.Y.append(y)
            for size in dtm.classification_data[2]: 
                self.sizes.append(size)


    def get_frame_dataset(self, kernel_size=64, 
                        augmentation=True, 
                        augmentation_steps=2, 
                        frame_steps=10, 
                        bbox_manipulation=2, 
                        min_width=8, 
                        outlier_detection=False,
                        x_shift=0, 
                        y_shift=0): 
        
        for dtm in self.dtm_list:
            dtm.set_matrix(dtm.resolution)
            dtm.get_bboxes()
            if outlier_detection == True: 
                dtm.outlier_detection(dtm.resolution)
                dtm.window_list = []
            dtm.set_matrix(dtm.resolution) # TODO: delete this line?
            dtm.set_label_matrix(bbox_manipulation=bbox_manipulation, min_width=min_width)
            dtm.create_frame_segmentation_data(kernel_size=kernel_size, augmentation=augmentation, 
                                               augmentation_steps=augmentation_steps, frame_steps=frame_steps, 
                                               x_shift=x_shift, y_shift=y_shift)
        self.X = []
        self.Y = []
        for dtm in self.dtm_list:
            for x in dtm.frame_segmentation_data[0]: 
                self.X.append(x)
            for y in dtm.frame_segmentation_data[1]: 
                self.Y.append(y)


    def get_mini_frame_dataset(self, kernel_size=64, 
                        augmentation=True, 
                        augmentation_steps=2, 
                        frame_steps=10, 
                        bbox_manipulation=2, 
                        min_width=8, 
                        outlier_detection=False,
                        x_shift=0, 
                        y_shift=0): 
        
        for dtm in self.dtm_list:
            dtm.set_matrix(dtm.resolution)
            dtm.get_bboxes()
            if outlier_detection == True: 
                dtm.outlier_detection(dtm.resolution)
                dtm.window_list = []
            dtm.set_matrix(dtm.resolution) # TODO: delete this line?
            dtm.set_label_matrix(bbox_manipulation=bbox_manipulation, min_width=min_width)
            dtm.create_mini_frame_segmentation_data(kernel_size=kernel_size, augmentation=augmentation, 
                                               augmentation_steps=augmentation_steps, frame_steps=frame_steps, 
                                               x_shift=x_shift, y_shift=y_shift)
        self.X = []
        self.Y = []
        for dtm in self.dtm_list:
            for x in dtm.frame_segmentation_data[0]: 
                self.X.append(x)
            for y in dtm.frame_segmentation_data[1]: 
                self.Y.append(y)



    def get_crxy_dataset(self, kernel_size=64, 
                        augmentation=True, 
                        augmentation_steps=2, 
                        frame_steps=10, 
                        bbox_manipulation=2, 
                        min_width=8, 
                        outlier_detection=False,
                        x_shift=0, 
                        y_shift=0): 
        
        for dtm in self.dtm_list:
            dtm.set_matrix(dtm.resolution)
            dtm.get_bboxes()
            if outlier_detection == True: 
                dtm.outlier_detection(dtm.resolution)
                dtm.window_list = []
            dtm.set_matrix(dtm.resolution) 
            dtm.set_crxy_label_matrix(bbox_manipulation=bbox_manipulation, min_width=min_width)
            dtm.create_crxy_data(kernel_size=kernel_size, augmentation=augmentation, 
                                 augmentation_steps=augmentation_steps, frame_steps=frame_steps, 
                                 x_shift=x_shift, y_shift=y_shift)
        self.X = []
        self.Y = []
        for dtm in self.dtm_list:
            for x in dtm.crxy_data[0]: 
                self.X.append(x)
            for y in dtm.crxy_data[1]: 
                self.Y.append(y)

    def create_rcnn_dataset(self, 
                            kernel_size=64, 
                            augmentation=True, 
                            augmentation_steps=2, 
                            frame_steps=10, 
                            data_type="train"): 
        from utils import matrix_smallsizing, get_frame_with_bbox_info
        """
        path = "numpy_matrices/Second_try"
        filename = "Steinkjer2-NDH_Steinkjer_5pkt_20175"
        file_format = "npy"
        xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20175shades.xml"
        dtm20 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)
        """
        index = 0
        for dtm in self.dtm_list:
            #dtm.set_matrix(dtm.resolution)  
            # self, csv_name, augmentation=False, augmentation_steps=-1, frame_size=64, frame_steps=12, filepath="", file_type="npy"   
            csv_name = "images/rcnn/" + dtm.filename + "_" + str(index) + ".csv"  
            filepath = "images/rcnn/npy_matrices/" + data_type + "/" + dtm.filename  
            dtm.set_matrix(dtm.resolution) 
            dtm.get_bboxes()
            dtm.create_detection_data(csv_name, filepath=filepath, frame_size=kernel_size, frame_steps=frame_steps, augmentation=augmentation, augmentation_steps=augmentation_steps)
            index += 1


    def create_rcnn_fullsize_dataset(self,                             
                            augmentation=True, 
                            augmentation_steps=2,                             
                            data_type="train"): 
        from utils import matrix_smallsizing, get_frame_with_bbox_info
        
        index = 0
        for dtm in self.dtm_list:  
            csv_name = "images/fullsize_rcnn/" + dtm.filename + "_" + str(index) + ".csv"  
            filepath = "images/fullsize_rcnn/npy_matrices/" + data_type + "/" + dtm.filename  
            dtm.set_matrix(dtm.resolution) 
            dtm.get_bboxes()
            dtm.create_fullsize_detection_data(csv_name, filepath=filepath, augmentation=augmentation, augmentation_steps=augmentation_steps)
            index += 1


    def create_pcr_dataset( self, 
                            kernel_size=64, 
                            augmentation=True, 
                            augmentation_steps=2, 
                            frame_steps=10, 
                            bbox_manipulation=2,                         
                            outlier_detection=False):                        
        
        for dtm in self.dtm_list:
            dtm.set_matrix(dtm.resolution)
            dtm.get_bboxes()

            if outlier_detection == True: 
                dtm.outlier_detection(dtm.resolution)
                dtm.window_list = []           
            
            dtm.create_pcr_data(augmentation=augmentation, 
                                augmentation_steps=augmentation_steps, 
                                frame_size=kernel_size, 
                                frame_steps=frame_steps)
            
        self.X = []
        self.Y_c = [] 
        self.Y_r = []
        for dtm in self.dtm_list:
            #for x in dtm.pcr_data[0]: 
                #self.X.append(x)
            self.X += dtm.pcr_data[0]
            #for yc in dtm.pcr_data[1]: 
            self.Y_c += dtm.pcr_data[1]
                #self.Y_c.append(yc)
            #for yr in dtm.pcr_data[2]: 
            self.Y_r += dtm.pcr_data[2]
                #self.Y_r.append(yr)
        print(len(self.X), len(self.Y_c), len(self.Y_r))

