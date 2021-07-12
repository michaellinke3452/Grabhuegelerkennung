import numpy as np 
import utils 
from dtm import DTM
from utils import get_mean_values  #DTM, 
import set_basic_trainingsset_without_ngm as sbt
import matplotlib.pyplot as plt

#DTM parameters: path, filename, file_format, resolution, xml_folder="", voc=False
path = "Forschungspraktikum/Grabhügeldetektor/Schweinert/pos_1"
filename = "DGMLASERSCAN_379617-5719970_dgm_KOPIE"
file_format = "xyz"
xml_folder = "Forschungspraktikum/Grabhügeldetektor/Schweinert/pos_1/Schweinert_100.xml"
dtm1 = DTM(path, filename, file_format, 1, xml_folder, voc=False) # 625 mounds


path = "Forschungspraktikum/Grabhügeldetektor/Pestrup/D27742_Linke"
filename = "dgm1_462522_5858087_dgm1_Kopie"
file_format = "xyz"
xml_folder = "Forschungspraktikum/Grabhügeldetektor/Pestrup_Schweinert/Pestruper_Graeberfeld_25.xml"
dtm2 = DTM(path, filename, file_format, 1, xml_folder, voc=False) # 499 mounds


path = "Forschungspraktikum/Grabhügeldetektor/Goldbeck/035-A-1108-2019_DGM1_Michael_Linke"
filename = "dgm1_542585_5916711_dgm1_KOPIE"
file_format = "xyz"
xml_folder = "Forschungspraktikum/Grabhügeldetektor/Pestrup_Schweinert/Goldbeck.xml"
dtm3 = DTM(path, filename, file_format, 1, xml_folder, voc=False) # 62 mounds


path = "numpy_matrices/Second_try"
filename = "Larvik-NDH_Larvik_5pkt_20170"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Larvik-NDH_Larvik_5pkt_20170shades.xml"
dtm4 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 42 mounds


path = "numpy_matrices/Second_try"
filename = "NDH_Åfjord_5pkt_20170"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/NDH_Afjord_5pkt_20170shades.xml"
dtm5 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 15 mounds


path = "numpy_matrices/Second_try"
filename = "NVE_Driva_20160"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/NVE_Driva_20160shades.xml"
dtm6 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 38 mounds


path = "numpy_matrices/Second_try"
filename = "NVE_Driva_20161"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/NVE_Driva_20161shades.xml"
dtm7 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 24 mounds


path = "numpy_matrices/Second_try"
filename = "Oppdal_12pkt_20111"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Oppdal_12pkt_20111shades.xml"
dtm8 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 95 mounds


path = "numpy_matrices/Second_try"
filename = "Oppdal_12pkt_20114"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Oppdal_12pkt_20114shades.xml"
dtm9 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 322 mounds


path = "numpy_matrices/Second_try"
filename = "Steinkjer_Skei-Hystad_20140"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Steinkjer_Skei-Hystad_20140shades.xml"
dtm10 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 30 mounds


path = "numpy_matrices/Second_try"
filename = "Steinkjer_Skei-Hystad_20141"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Steinkjer_Skei-Hystad_20141shades.xml"
dtm11 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False) # 9 mounds



path = "images"
filename = "vieritz"
file_format = "npy"
xml_folder = "images/vieritz.xml"
dtm21 = DTM(path, filename, file_format, 1, xml_folder, voc=False) # 5 mounds



dtm_list = [dtm1, dtm2, dtm3, dtm4, dtm5, dtm6, dtm7, dtm8, dtm9, dtm10, dtm11, dtm21]
window_list = sbt.generate_DTM_training_data(dtm_list, 2, as_shade=False)
print(len(window_list))
get_mean_values(window_list)
right_sized_windows = [] 


size_list = []
for w in window_list: 
    size_list.append(int(np.median(w.shape) + 0.5))
size_list = np.asarray(size_list) 
#print(len(window_list), len(size_list))
print(size_list[-20:-1])

"""
for w in window_list: 
    if np.min(w.shape) >= 5 and np.max(w.shape) < 30: 
        right_sized_windows.append(w) 
print(len(right_sized_windows))

"""

np.save("full_raw_gm_data_with_vieritz.npy", right_sized_windows)
np.save("full_raw_gm_data_with_vieritz_sizes.npy", size_list)
"""

path = "numpy_matrices/Second_try"
filename = "Steinkjer-NDH_Steinkjer_5pkt_20170"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Steinkjer-NDH_Steinkjer_5pkt_20170shades.xml"
dtm12 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

path = "numpy_matrices/Second_try"
filename = "Steinkjer-NDH_Steinkjer_5pkt_20171"
file_format = "npy"
xml_folder = "images/shade_png/shade_labels/Steinkjer-NDH_Steinkjer_5pkt_20171shades.xml"
dtm12 = DTM(path, filename, file_format, 0.25, xml_folder, voc=False)

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





ngm_dtm_list = [dtm12, dtm13, dtm14, dtm15, dtm16, dtm17, dtm18, dtm19, dtm20]

window_list = sbt.generate_DTM_training_data(ngm_dtm_list, 2, as_shade=False)
print(len(window_list))
get_mean_values(window_list)

np.save("ngm_train_data_norway.npy", window_list) 
"""

"""
for i in range(10):
    plt.imshow(right_sized_windows[i])
    plt.show()
"""

"""
x = np.arange(1, 51)
window_list1 = sbt.generate_DTM_training_data([dtm1], 2, as_shade=False)
y1 = get_mean_values(window_list1)
plt.plot(x, y1) 
plt.show()

window_list2 = sbt.generate_DTM_training_data([dtm2], 2, as_shade=False)
y2 = get_mean_values(window_list2)
plt.plot(x, y2) 
plt.show()

window_list3 = sbt.generate_DTM_training_data([dtm3], 2, as_shade=False)
y3 = get_mean_values(window_list3)
plt.plot(x, y3) 
plt.show()

res = 2

window_list4 = sbt.generate_DTM_training_data([dtm4], res, as_shade=False)
y4 = get_mean_values(window_list4)
plt.plot(x, y4) 
plt.show()

window_list5 = sbt.generate_DTM_training_data([dtm5], res, as_shade=False)
y5 = get_mean_values(window_list5)
plt.plot(x, y5) 
plt.show()

window_list6 = sbt.generate_DTM_training_data([dtm6], res, as_shade=False)
y6 = get_mean_values(window_list6)
plt.plot(x, y6) 
plt.show()

window_list7 = sbt.generate_DTM_training_data([dtm7], res, as_shade=False)
y7 = get_mean_values(window_list7)
plt.plot(x, y7) 
plt.show()

window_list8 = sbt.generate_DTM_training_data([dtm8], res, as_shade=False)
y8 = get_mean_values(window_list8)
plt.plot(x, y8) 
plt.show()

window_list9 = sbt.generate_DTM_training_data([dtm9], res, as_shade=False)
y9 = get_mean_values(window_list9)
plt.plot(x, y9) 
plt.show()

window_list10 = sbt.generate_DTM_training_data([dtm10], res, as_shade=False)
y10 = get_mean_values(window_list10)
plt.plot(x, y10) 
plt.show()

window_list11 = sbt.generate_DTM_training_data([dtm11], res, as_shade=False)
y11 = get_mean_values(window_list11)
plt.plot(x, y11) 
plt.show()


x = np.arange(1, 51)
for y in [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11]: 
    plt.plot(x, y, 'o') 
plt.show()

all_windows = [window_list1, window_list2, window_list3, window_list4, window_list5, window_list6, window_list7, window_list8, window_list9, window_list10, window_list11]
num = 0 
for w in all_windows: 
    num += len(w) 
print(num)
"""




"""
for w in window_list2: 
    plt.imshow(w)
    plt.show()
"""    