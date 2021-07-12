# builds datasets with training data unapproved for publication, 
# test data is open.

import numpy as np 
#import utils 
from dtm import DTM, DTMList
from preprocessing import get_mean_values  #DTM, 
import set_basic_trainingsset_without_ngm as sbt
import matplotlib.pyplot as plt
import sys


def get_meta_infos():
    #DTM parameters: path, filename, file_format, resolution, xml_folder="", voc=False
    path = "Forschungspraktikum/Grabhügeldetektor/Schweinert/pos_1"
    filename = "DGMLASERSCAN_379617-5719970_dgm_KOPIE"
    file_format = "xyz"
    xml_folder = "Forschungspraktikum/Grabhügeldetektor/Schweinert/pos_1/Schweinert_100.xml"
    dtm1 = DTM(path, filename, file_format, 1, xml_folder, voc=False)

    
    path = "Forschungspraktikum/Grabhügeldetektor/Pestrup/D27742_Linke"
    filename = "dgm1_462522_5858087_dgm1_Kopie"
    file_format = "xyz"
    xml_folder = "Forschungspraktikum/Grabhügeldetektor/Pestrup_Schweinert/Pestruper_Graeberfeld_25.xml"
    dtm2 = DTM(path, filename, file_format, 1, xml_folder, voc=False)


    path = "Forschungspraktikum/Grabhügeldetektor/Goldbeck/035-A-1108-2019_DGM1_Michael_Linke"
    filename = "dgm1_542585_5916711_dgm1_KOPIE"
    file_format = "xyz"
    xml_folder = "Forschungspraktikum/Grabhügeldetektor/Pestrup_Schweinert/Goldbeck.xml"
    dtm3 = DTM(path, filename, file_format, 1, xml_folder, voc=False)
    

    path = "numpy_matrices/Second_try"
    filename = "Larvik-NDH_Larvik_5pkt_20170"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Larvik-NDH_Larvik_5pkt_20170shades.xml"
    dtm4 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "NDH_Åfjord_5pkt_20170"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/NDH_Afjord_5pkt_20170shades.xml"
    dtm5 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "NVE_Driva_20160"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/NVE_Driva_20160shades.xml"
    dtm6 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "NVE_Driva_20161"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/NVE_Driva_20161shades.xml"
    dtm7 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "Oppdal_12pkt_20111"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Oppdal_12pkt_20111shades.xml"
    dtm8 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "Oppdal_12pkt_20114"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Oppdal_12pkt_20114shades.xml"
    dtm9 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "Steinkjer_Skei-Hystad_20140"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer_Skei-Hystad_20140shades.xml"
    dtm10 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)


    path = "numpy_matrices/Second_try"
    filename = "Steinkjer_Skei-Hystad_20141"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer_Skei-Hystad_20141shades.xml"
    dtm11 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)



    path = "images"
    filename = "vieritz"
    file_format = "npy"
    xml_folder = "images/vieritz.xml"
    dtm21 = DTM(path, filename, file_format, 1, xml_folder, voc=False)



    path = "numpy_matrices/Second_try"
    filename = "Steinkjer-NDH_Steinkjer_5pkt_20170"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer-NDH_Steinkjer_5pkt_20170shades.xml"
    dtm12 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer-NDH_Steinkjer_5pkt_20171"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer-NDH_Steinkjer_5pkt_20171shades.xml"
    dtm12a = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer-NDH_Steinkjer_5pkt_20172"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer-NDH_Steinkjer_5pkt_20172shades.xml"
    dtm13 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer-NDH_Steinkjer_5pkt_20173"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer-NDH_Steinkjer_5pkt_20173shades.xml"
    dtm14 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer2-NDH_Steinkjer_5pkt_20170"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20170shades.xml"
    dtm15 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer2-NDH_Steinkjer_5pkt_20171"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20171shades.xml"
    dtm16 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer2-NDH_Steinkjer_5pkt_20172"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20172shades.xml"
    dtm17 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer2-NDH_Steinkjer_5pkt_20173"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20173shades.xml"
    dtm18 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer2-NDH_Steinkjer_5pkt_20174"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20174shades.xml"
    dtm19 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    path = "numpy_matrices/Second_try"
    filename = "Steinkjer2-NDH_Steinkjer_5pkt_20175"
    file_format = "npy"
    xml_folder = "images/shade_png/shade_labels/Steinkjer2-NDH_Steinkjer_5pkt_20175shades.xml"
    dtm20 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

    return dtm1, dtm2, dtm3, dtm4, dtm5, dtm6, dtm7, dtm8, dtm9, dtm10, dtm11, dtm12, dtm12a, dtm13, dtm14, dtm15, dtm16, dtm17, dtm18, dtm19, dtm20, dtm21




def get_DTM_pcr_data(bbox_manipulation=2, outlier_detection=False):


    dtm1, dtm2, dtm3, dtm4, dtm5, dtm6, dtm7, dtm8, dtm9, dtm10, dtm11, dtm12, dtm12a, dtm13, dtm14, dtm15, dtm16, dtm17, dtm18, dtm19, dtm20, dtm21 = get_meta_infos()



    dtm_germany_train = DTMList([dtm1, dtm2, dtm3]) 
    dtm_germany_test = DTMList([dtm21])  

    dtm_norway_train = DTMList([dtm4, dtm8, dtm9])
    dtm_norway_test = DTMList([dtm5, dtm6, dtm7, dtm10, dtm11])

    dtm_norway_1 = DTMList([dtm12, dtm12a, dtm13, dtm14, dtm15, dtm16, dtm17, dtm18, dtm19, dtm20])

    augmentation_steps = 2
    frame_steps = 20
    dtm_germany_train.create_pcr_dataset(   kernel_size=64, 
                                            augmentation=True, 
                                            augmentation_steps=augmentation_steps, 
                                            frame_steps=frame_steps * augmentation_steps, 
                                            bbox_manipulation=bbox_manipulation,                                             
                                            outlier_detection=outlier_detection)

    dtm_germany_test.create_pcr_dataset(    kernel_size=64, 
                                            augmentation=True, 
                                            augmentation_steps=augmentation_steps, 
                                            frame_steps=frame_steps * augmentation_steps, 
                                            bbox_manipulation=bbox_manipulation,                                             
                                            outlier_detection=outlier_detection)

    augmentation_steps = 8
    frame_steps = 20
    dtm_norway_train.create_pcr_dataset(    kernel_size=64, 
                                            augmentation=True, 
                                            augmentation_steps=augmentation_steps, 
                                            frame_steps=frame_steps * augmentation_steps, 
                                            bbox_manipulation=bbox_manipulation,                                             
                                            outlier_detection=outlier_detection)


    dtm_norway_test.create_pcr_dataset(     kernel_size=64, 
                                            augmentation=True, 
                                            augmentation_steps=augmentation_steps, 
                                            frame_steps=frame_steps * augmentation_steps, 
                                            bbox_manipulation=bbox_manipulation,                                             
                                            outlier_detection=outlier_detection)

    augmentation_steps = 2
    frame_steps = 20
    dtm_norway_1.create_pcr_dataset(        kernel_size=64, 
                                            augmentation=True, 
                                            augmentation_steps=augmentation_steps, 
                                            frame_steps=frame_steps * augmentation_steps, 
                                            bbox_manipulation=bbox_manipulation,                                             
                                            outlier_detection=outlier_detection  )

    
    dtm_norway_train.X = dtm_norway_train.X + dtm_norway_1.X 
    dtm_norway_train.Y_c = dtm_norway_train.Y_c + dtm_norway_1.Y_c
    dtm_norway_train.Y_r = dtm_norway_train.Y_r + dtm_norway_1.Y_r


    return [[dtm_germany_train, dtm_germany_test], [dtm_norway_train, dtm_norway_test]]

