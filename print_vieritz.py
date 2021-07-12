import numpy as np 
#import utils 
from dtm import DTM
from preprocessing import transform  
#import set_basic_trainingsset_without_ngm as sbt
import matplotlib.pyplot as plt
import sys



path = "images"
filename = "vieritz"
file_format = "npy"
xml_folder = "images/vieritz.xml"
dtm = DTM(path, filename, file_format, 1, xml_folder, voc=False)


dtm.set_matrix(dtm.resolution)
dtm.get_bboxes()
dtm.set_label_matrix(bbox_manipulation=0, min_width=1)

matrix = transform(dtm.matrix, "laplacian")
matrix = np.clip(matrix, -1, 1)
plt.subplot(1,2,1)
plt.imshow(matrix, cmap="Greys") 
plt.subplot(1,2,2) 
plt.imshow(dtm.label_matrix, cmap="hot")      
plt.show()