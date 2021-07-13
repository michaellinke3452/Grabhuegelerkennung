import numpy as np
from bounding_box import BBox
import matplotlib.pyplot as plt


def add_bbox_to_matrix(matrix, bbox):     
    matrix[bbox.xmin: bbox.xmax, bbox.ymin: bbox.ymax] += 1
    return matrix


def set_matrix(matrix, boxes): 
    for bbox in boxes: 
        matrix = add_bbox_to_matrix(matrix, bbox) 
    matrix[matrix > 0.5] = 1
    return matrix


def build_eval_matrix(xsize, ysize, true_boxes, pred_boxes): 
    matrix1 = np.zeros((xsize, ysize)) 
    matrix2 = np.zeros((xsize, ysize)) 
    matrix1 = set_matrix(matrix1, true_boxes) 
    matrix2 = set_matrix(matrix2, pred_boxes)    
    return matrix1 + matrix2


def iou(matrix): 
    union = 0 
    intersection = 0 
    for i in range(matrix.shape[0]): 
        for j in range(matrix.shape[1]): 
            if matrix[i][j] == 1 or matrix[i][j] == 2:
                union += 1 
            if matrix[i][j] == 2:
                intersection += 1 
    if union == 0: 
        return 0 
    else: 
        return intersection / union    


def iou_over_matrix_list(true_matrices, pred_matrices): 
    # for every matrix, the set_matrix() operation has been done already. 
    union = 0 
    intersection = 0   

    for m in range(len(true_matrices)):
        matrix = true_matrices[m] + pred_matrices[m] 
        for i in range(matrix.shape[0]): 
            for j in range(matrix.shape[1]): 
                if matrix[i][j] == 1 or matrix[i][j] == 2:
                    union += 1 
                if matrix[i][j] == 2:
                    intersection += 1 
    if union == 0: 
        return 0 
    else: 
        return intersection / union



"""
bbox = BBox(25, 75, 25, 75, "", "", ID=0)
#print(bbox.get_info())
matrix = np.zeros((100, 100)) 

add_bbox_to_matrix(matrix, bbox)
bbox = BBox(30, 80, 30, 80, "", "", ID=1)
add_bbox_to_matrix(matrix, bbox)

print(iou(matrix))

plt.imshow(matrix) 
plt.show()
"""