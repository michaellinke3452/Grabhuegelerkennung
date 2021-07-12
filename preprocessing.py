import numpy as np
import matplotlib.pyplot as plt
import sys

"""
def standard_scaling(data_list): 
    from sklearn.preprocessing import StandardScaler 
    for i in range(len(data_list)): 
        scaler = StandardScaler() 
        scaler.fit(data_list[i]) 
        data_list[i] = scaler.transform(data_list[i])
"""

def standard_scaling(data_list): 
    for i in range(len(data_list)): 
        data_mean = np.mean(data_list[i]) 
        data_std = np.std(data_list[i])         
        data_list[i] = (data_list[i] - data_mean) / data_std
    return data_list


def normalize(data_list): 
    for i in range(len(data_list)): 
        data_list[i] -= np.min(data_list[i]) 
        data_list[i] /= np.max(data_list[i]) 

def rotate_matrix(matrix): 
    m1 = np.rot90(matrix, 1) 
    m2 = np.rot90(matrix, 2) 
    m3 = np.rot90(matrix, 3)
    return [matrix, m1, m2, m3]


def rotate_dataset(dataset, labels): 
    X = [] 
    Y = []
    for d in range(len(dataset)): 
        x = rotate_matrix(dataset[d]) 
        y = rotate_matrix(labels[d])
        for i in range(len(x)): 
            X.append(x[i]) 
            Y.append(y[i])
    return X, Y

def rotate_classification_dataset(dataset, labels): 
    X = [] 
    Y = []
    for d in range(len(dataset)): 
        x = rotate_matrix(dataset[d]) 
        #y = [labels[d], labels[d], labels[d], labels[d]] 
        for i in range(len(x)): 
            X.append(x[i]) 
            Y.append(labels[d])
    return X, Y





def show_matrix(matrix): 
    plt.imshow(matrix * (-1), cmap="Greys") 
    plt.show()

def get_shade_matrix(matrix): 
    shade_matrix = matrix.copy()
    for i in range(matrix.shape[0]): 
        for j in range(1, matrix.shape[1]): 
            shade_matrix[i][j] = matrix[i][j] - matrix[i][j-1] 
    shade_matrix = np.clip(shade_matrix, -0.2, 0.2)
    return shade_matrix

def shade_view(matrix): 
    shade_matrix = matrix.copy()
    for i in range(matrix.shape[0]): 
        for j in range(1, matrix.shape[1]): 
            shade_matrix[i][j] = matrix[i][j] - matrix[i][j-1] 
    
    shade_matrix2 = matrix.copy()
    for i in range(matrix.shape[1]): 
        for j in range(1, matrix.shape[0]): 
            shade_matrix2[j][i] = matrix[j][i] - matrix[j - 1][i] 
    shade_matrix += (0.5 * shade_matrix2)
    shade_matrix *= (-1)
    shade_matrix = np.clip(shade_matrix, -0.2, 0.2)
    plt.imshow(shade_matrix, cmap="Greys") 
    plt.show()



def shade_save(matrix, filename): 
    shade_matrix = matrix.copy()
    for i in range(matrix.shape[0]): 
        for j in range(1, matrix.shape[1]): 
            shade_matrix[i][j] = matrix[i][j] - matrix[i][j-1]     
    shade_matrix2 = matrix.copy()
    for i in range(matrix.shape[1]): 
        for j in range(1, matrix.shape[0]): 
            shade_matrix2[j][i] = matrix[j][i] - matrix[j - 1][i] 
    shade_matrix += (0.5 * shade_matrix2)
    shade_matrix *= (-1)
    print(np.min(shade_matrix), np.max(shade_matrix))
    shade_matrix = np.clip(shade_matrix, -0.2, 0.2)
    np.save(filename, shade_matrix)


def write_shade_as_png(m, filename):
    from PIL import Image   
    matrix = m.copy()
    x_max = matrix.shape[0]
    y_max = matrix.shape[1]
    matrix -= np.min(matrix) 
    matrix /= np.max(matrix) 
    matrix *= 255
    matrix_as_image = np.zeros((x_max, y_max), dtype=np.uint8)
    for x in range(0, x_max):
        for y in range(0, y_max):            
            matrix_as_image[x][y] = matrix[x][y]
            
    img = Image.fromarray(matrix_as_image, mode="L")
    img.save(filename)
 


def open_numpy_matrices(func): 
    import glob    
    for filename in glob.glob("numpy_matrices/Second_try/" + "/*.npy"): 
        matrix = np.load(filename)
        func(matrix, filename.replace(".npy", "shades.npy"))
        

def shades_to_png(): 
    import glob     
    for filename in glob.glob("numpy_matrices/shades/" + "/*.npy"): 
        matrix = np.load(filename)
        write_shade_as_png(matrix, filename.replace(".npy", ".png"))



def get_mean_values(window_list): 
    x = np.arange(1, 51)
    y = np.zeros(50)
    num = np.zeros(50)
    for w in window_list: 
        d = int(np.mean(w.shape))
        h = np.max(w) 
        if d <= 50:
            y[d] += h 
            num[d] += 1 
    for i in range(50):
        if num[i] != 0:  
            y[i] /= num[i] 
    return y



def matrix_derivative_2d(matrix, mode="raw"): 
    # the horizontal difference is concidered to be 1.
    # modes: angle, percentage, raw
    
    der_x = (matrix - np.roll(matrix, -1, axis=1))     
    der_y = (matrix - np.roll(matrix, 1, axis=0)) 
    der = np.sqrt(der_x ** 2 + der_y ** 2)
    if mode == "raw":
        return der
    elif mode == "percentage": 
        return 100 * der
    elif mode == "angle": 
        return np.arctan(der)
    else: 
        raise Exception("No valid mode! Choose raw, percentage or angle!")

def matrix_derivative_2d_one_dir(matrix, direction="x", mode="raw"): 
    # the horizontal difference is concidered to be 1.
    if direction == "x":
        der_x = matrix - np.roll(matrix, -1, axis=1)
        der = der_x
    elif direction == "y":
        der_y = matrix - np.roll(matrix, 1, axis=0)   
        der = der_y 
    else: 
        raise Exception("No valid direction! x or y has to be chosen!")
        return -1
    if mode == "raw":
        return der
    elif mode == "percentage": 
        return 100 * der
    elif mode == "angle": 
        return np.arctan(der)
    else: 
        raise Exception("No valid mode! Choose raw, percentage or angle!")
    

def aspect_2d(matrix): 
    der_x = (matrix - np.roll(matrix, -1, axis=1))  
    der_y = (matrix - np.roll(matrix, 1, axis=0))
    der_y[der_y == 0] = 10e-9
    return np.arctan2(der_x, der_y)


def get_hillshade(matrix, azimuth, altitude, slope, aspect):
    from numpy import pi, cos, sin

    slope = matrix_derivative_2d(matrix, mode="angle") 
    aspect = aspect_2d(matrix) 

    a = azimuth 
    h = altitude 
    z = np.pi / 2. - h 

    hillshade = cos(z) * cos(slope) + sin(z) * sin(slope) * cos(aspect - a)

    return hillshade 


def cv2_blur(matrix, ksize): 
    # ksize: tuple of two integers.
    from cv2 import blur 
    return blur(matrix, ksize) 
    


def slrm_cv2_average(matrix, ksize): 
    # ksize: tuple of two integers.
    from cv2 import blur 
    blurred = blur(matrix, ksize) 
    return matrix - blurred 

def slrm_cv2_gaussian(matrix, ksize): 
    # ksize: tuple of two integers.
    from cv2 import GaussianBlur 
    blurred = GaussianBlur(matrix, ksize, sigmaX=0) 
    return matrix - blurred 


def slrm_cv2_median(matrix, ksize): 
    # ksize: tuple of two integers.
    from cv2 import medianBlur 
    blurred = medianBlur(matrix, ksize) 
    return matrix - blurred 

def slrm_cv2_bilateral(matrix, ksize): 
    # ksize: tuple of two integers.
    from cv2 import bilateralFilter 
    blurred = bilateralFilter(matrix, ksize, 75, 75) 
    return matrix - blurred

def cv2_laplacian(matrix): 
    from cv2 import Laplacian, CV_32F    
    return Laplacian(matrix, CV_32F)


def cv2_gauss_laplacian(matrix): 
    from cv2 import Laplacian, CV_32F, GaussianBlur
    blurred = GaussianBlur(matrix, (41, 41), sigmaX=0)
    return Laplacian(blurred, CV_32F)    


def cv2_filter2D(matrix): 
    from cv2 import filter2D, Laplacian, CV_32F
    #kernel = np.array([[1,0,0], [1,1,0], [1,1,1]])
    #kernel = np.array([[0,1,0], [1,1,1], [0,1,0]])
    kernel = np.array([[1,0,1], [0,1,0], [1,0,1]])
    blurred = filter2D(matrix, -1, kernel)
    return Laplacian(blurred, CV_32F)

def cv2_relief(matrix): 
    # TODO: Find citation for relief filter!! 
    from cv2 import filter2D, Laplacian, CV_32F
    kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
    blurred = filter2D(matrix, -1, kernel)
    return Laplacian(blurred, CV_32F)

"""
def cv2_sobel2D(matrix): 
    from cv2 import Sobel, CV_32F 
    x = Sobel(matrix, CV_32F, 1, 0, ksize=5)
    y = Sobel(matrix, CV_32F, 0, 1, ksize=5)
    return np.sqrt(x ** 2 + y ** 2)
"""
def cv2_sobel2D(matrix): 
    from cv2 import Sobel, CV_64F 
    x = Sobel(matrix, CV_64F, 1, 0, ksize=5)
    y = Sobel(matrix, CV_64F, 0, 1, ksize=5)
    return np.sqrt(x ** 2 + y ** 2)

def cv2_gauss_laplacian_slrm(matrix): 
    blurred = cv2_gauss_laplacian(matrix) 
    return matrix - blurred

def cv2_relief_laplacian_slrm(matrix): 
    blurred = cv2_relief(matrix) 
    return matrix - blurred

def double_sobel(matrix): 
    matrix = cv2_sobel2D(matrix) 
    return cv2_sobel2D(matrix)

def double_slope(matrix): 
    matrix = matrix_derivative_2d(matrix, mode="angle") 
    return matrix_derivative_2d(matrix, mode="angle") 

def double_slope_one_dir(matrix, direction): 
    matrix = matrix_derivative_2d_one_dir(matrix, direction=direction, mode="angle") 
    return matrix_derivative_2d_one_dir(matrix, direction=direction, mode="angle") 

def double_slope2(matrix): 
    matrix1 = double_slope_one_dir(matrix, "x") 
    matrix2 = double_slope_one_dir(matrix, "y") 
    return np.sqrt(matrix1**2 + matrix2**2)

def aspect_sobel(matrix): 
    matrix = aspect_2d(matrix) 
    return cv2_sobel2D(matrix)

def sobel_laplacian(matrix): 
    matrix = cv2_sobel2D(matrix)
    return cv2_laplacian(matrix)

def pca(matrix, num_components): 
    from sklearn.decomposition import PCA 
    pca_instance = PCA(num_components) 
    width = matrix.shape[0] 
    height = matrix.shape[1] 
    #f_matrix = np.reshape(matrix, (-1, width * height))
    f_matrix = matrix
    decomposed = pca_instance.fit_transform(f_matrix)
    inverted = pca_instance.inverse_transform(decomposed) 
    return inverted





"""
The following functions are implementations of the equations 4 - 28 in: 
Florinsky, I. V. (2017). An illustrated introduction to general geomorphometry. Progress in Physical Geography, 41(6), 723-752.
The numbers in brackets refer to the equation number in the article.
"""




def numpy_gradient(matrix): 
    return np.gradient(matrix)

def get_gradients(matrix): 
    #print(matrix.shape)
    z = np.gradient(matrix) 
    p = z[0] 
    q = z[1] 
    grad_p = np.gradient(p) 
    r = grad_p[0] 
    s = grad_p[1] 
    grad_q = np.gradient(q) 
    s2 = grad_q[0] 
    t = grad_q[1]     
    return p, q, r, s, t


def get_first_gradients(matrix): 
    #print(matrix.shape)
    z = np.gradient(matrix) 
    p = z[0] 
    q = z[1]        
    return p, q

def get_second_gradients(matrix): 
    #print(matrix.shape)
    z = np.gradient(matrix) 
    p = z[0] 
    q = z[1]  
    grad_p = np.gradient(p) 
    r = grad_p[0]     
    grad_q = np.gradient(q) 
    t = grad_q[1]     
    return r, t      
    



def get_third_degree_gradients(r, t): 
    grad_r = np.gradient(r) 
    g = grad_r[0] 
    k = grad_r[1] 
    grad_t = np.gradient(t) 
    m = grad_t[0] 
    h = grad_t[1] 
    return g, h, k, m 


def slope(p, q): 
    # (4)
    G = np.sqrt(p**2 + q**2) 
    G = np.nan_to_num(G)    
    return np.nan_to_num(np.arctan(G))  


def aspect(p, q): 
    # (5)
    A = - 90 * (1 - np.sign(q)) * (1 - np.abs(np.sign(p))) 
    A += 180 * (1 + np.sign(p)) 
    A -= (180 / np.pi) * np.sign(p) * np.arccos(-q / np.sqrt(p**2 + q**2)) 
    A = np.nan_to_num(A)
    return A 


def northwardness(A): 
    # (6)
    return np.cos(A) 


def eastwardness(A): 
    # (7)
    return np.sin(A) 


def plan_curvature(p, q, r, s, t):
    # (8) 
    numerator = q**2 * r - 2*p*q*s + p**2 * t  
    numerator *= (-1) 
    denominator = np.sqrt((p**2 + q**2)**3) 
    return np.nan_to_num(numerator / denominator)  


def horizontal_curvature(p, q, r, s, t): 
    # (9)
    numerator = q**2 * r - 2*p*q*s + p**2 * t 
    numerator *= (-1) 
    denominator = (p**2 + q**2) * np.sqrt(1 + p**2 + q**2) 
    return np.nan_to_num(numerator / denominator)  


def vertical_curvature(p, q, r, s, t): 
    # (10)
    numerator = p**2 * r + 2*p*q*s + q**2 * t 
    numerator *= (-1) 
    denominator = (p**2 + q**2) * np.sqrt((1 + p**2 + q**2)**3) 
    A = numerator / denominator 
    A = np.nan_to_num(A)
    return A


def difference_curvature(p, q, r, s, t): 
    # (11)
    return np.nan_to_num((1/2) * (vertical_curvature(p, q, r, s, t) - horizontal_curvature(p, q, r, s, t)) ) 


def accumulation_curvature(p, q, r, s, t): 
    # (14)
    return np.nan_to_num(horizontal_curvature(p, q, r, s, t) * vertical_curvature(p, q, r, s, t))  


def ring_curvature(p, q, r, s, t): 
    # (15)
    numerator = (p**2 - q**2) * s - p*q*(r - t) 
    denominator = (p**2 + q**2) * (1 + p**2 + q**2) 
    return np.nan_to_num((numerator / denominator)**2)  


def rotor(p, q, r, s, t): 
    # (16)
    numerator = (p**2 - q**2) * s - p*q*(r - t) 
    denominator = np.sqrt((p**2 + q**2)**3) 
    return np.nan_to_num(numerator / denominator)  

 
def horizontal_curvature_deflection(p, q, r, s, t, g, h, k, m): 
    # (17)
    D = q**3 * g - p**3 * h + 3*p*q*(p*m - q*k) 
    D /= np.sqrt((p**2 + q**2)**3 * (1 + p**2 + q**2)) 
    T = 2 + 3*(p**2 + q**2) 
    T /= (1 + p**2 + q**2)    
    D -= horizontal_curvature(p, q, r, s, t) * rotor(p, q, r, s, t) * T 
    return np.nan_to_num(D)  


def vertical_curvature_deflection(p, q, r, s, t, g, h, k, m): 
    # (18)
    D = q**3 * m - p**3 * k + 2*p*q*(q*k - p*m) - p*q*(q*h - p*g) 
    D /= np.sqrt( (p**2 + q**2)**3 * (1 + p**2 + q**2)**3) 
    a = 2 * (r + t) / np.sqrt((1 + p**2 + q**2)**3) 
    b = (2 + 5*(p**2 + q**2)) / (1 + p**2 + q**2)
    D -= rotor(p, q, r, s, t) * (a + vertical_curvature(p, q, r, s, t) * b)
    return np.nan_to_num(D)  


def mean_curvature(p, q, r, s, t): 
    # (21)
    numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
    denominator = 2 * np.sqrt((1 + p**2 + q**2)**3)     
    return np.nan_to_num(numerator / denominator)            

def gaussian_curvature(p, q, r, s, t): 
    # (22)
    numerator = r*t - s**2 
    denominator = (1 + p**2 + q**2)**2 
    return np.nan_to_num(numerator / denominator) 


def minimal_curvature(p, q, r, s, t): 
    # (19)
    H = mean_curvature(p, q, r, s, t)
    K = gaussian_curvature(p, q, r, s, t)
    return np.nan_to_num(H - np.sqrt(H**2 - K)) 


def maximal_curvature(p, q, r, s, t): 
    # (20)
    H = mean_curvature(p, q, r, s, t)
    K = gaussian_curvature(p, q, r, s, t)
    return np.nan_to_num(H + np.sqrt(H**2 - K)) 


def unsphericity_curvature(p, q, r, s, t):
    # (23)
    H = mean_curvature(p, q, r, s, t)
    K = gaussian_curvature(p, q, r, s, t)
    return np.nan_to_num(np.sqrt(H**2 - K)) 


def horizontal_excess_curvature(p, q, r, s, t): 
    # (12)
    M = unsphericity_curvature(p, q, r, s, t)
    E = difference_curvature(p, q, r, s, t)
    A = M - E
    A = np.nan_to_num(A)
    return A


def vertical_excess_curvature(p, q, r, s, t): 
    # (13)
    M = unsphericity_curvature(p, q, r, s, t)
    E = difference_curvature(p, q, r, s, t)
    return np.nan_to_num(M + E) 


def laplacian(r, t): 
    # (24)
    return np.nan_to_num(r + t)  


def shape_index(p, q, r, s, t): 
    # (25)
    k_min = minimal_curvature(p, q, r, s, t) 
    k_max = maximal_curvature(p, q, r, s, t) 
    return np.nan_to_num((2/np.pi) * np.arctan((k_max + k_min) / (k_max - k_min)))  


def curvedness(p, q, r, s, t): 
    # (26)
    k_min = minimal_curvature(p, q, r, s, t) 
    k_max = maximal_curvature(p, q, r, s, t) 
    C = (k_max**2 + k_min**2) / 2 
    return np.nan_to_num(np.sqrt(C))  

def cot(x): 
    # cotangent
    return np.nan_to_num(np.cos(x) / np.sin(x))  


def reflectance(p, q, a, e): 
    # (27)
    # a = azimuth 
    # e = elevation
    from numpy import sin, cos
    R = 1 - p*sin(a) * cot(e) - q*cos(a) * cot(e) 
    R /= (np.sqrt(1 + p**2 + q**2) * np.sqrt(1 + (sin(a) * cot(e)**2 + (cos(a) * cot(e))))) 
    return np.nan_to_num(R)  


def insolation(p, q, a, e): 
    # (28)
    # a = azimuth 
    # e = elevation
    from numpy import sin, cos
    I = 50 * ( 1 + np.sign(  sin(e) - cos(e) * (p*sin(a) + q*cos(a)))) 
    I *= ((sin(e) - cos(e) * (p*sin(a) + q*cos(a)))  /  (np.sqrt(1 + p**2 + q**2))) 
    return np.nan_to_num(I)   



# selfmade stuff based on morphological filters




def truncated_laplacian(r, t): 
    tl = laplacian(r, t) 
    tl[tl<0] = -0.0000001 
    return tl 

def truncated_minimal_curvature(p, q, r, s, t): 
    tmc = minimal_curvature(p, q, r, s, t)
    tmc[tmc<0] = -0.0000001 
    return tmc


def truncated_usc(p, q, r, s, t): 
    tusc = unsphericity_curvature(p, q, r, s, t)
    tusc[tusc<0] = -0.0000001 
    return tusc

def mean_vertical_minimal_curvature(p, q, r, s, t):
    
    ve = vertical_curvature(p, q, r, s, t)    
    mi = minimal_curvature(p, q, r, s, t)
    me = mean_curvature(p, q, r, s, t)    
    return (ve + mi + me) / 3.



def altered_unsphericity_curvature(p, q, r, s, t):
    
    H = vertical_curvature(p, q, r, s, t)    
    K = minimal_curvature(p, q, r, s, t)
    #K -= np.min(K)
    print(np.min(H), np.max(H))
    print(np.min(K), np.max(K))
    #return np.sqrt(K**2 - H)    
    return np.sqrt(H**2 - K)

def altered_unsphericity_curvature2(p, q, r, s, t):
    
    H = vertical_curvature(p, q, r, s, t)
    K = minimal_curvature(p, q, r, s, t)
    M = mean_curvature(p, q, r, s, t)
    #return np.sqrt(K**2 - H)    
    #return np.sqrt(H**2 - K)
    return np.sqrt(M**2 + H**2 - K)

def altered_unsphericity_curvature3(p, q, r, s, t):
    
    H = vertical_curvature(p, q, r, s, t)
    return slrm_cv2_average(H, (41, 41))
    #K = minimal_curvature(p, q, r, s, t)
    #M = mean_curvature(p, q, r, s, t)
    #return np.sqrt(K**2 - H)    
    #return np.sqrt(H**2 - K)
    #return np.sqrt(M**2 + H**2 - K)

def altered_unsphericity_curvature4(p, q, r, s, t):
    
    #H = mean_curvature(p, q, r, s, t)    
    #K = gaussian_curvature(p, q, r, s, t)
    H = vertical_curvature(p, q, r, s, t)    
    K = minimal_curvature(p, q, r, s, t)
    #K = curvedness(p, q, r, s, t)
    #K -= np.min(K)
    print(np.min(H), np.max(H))
    print(np.min(K), np.max(K))
    #return np.sqrt(K**2 - H)    
    return - np.log(np.sqrt(H**2 + K**2))
"""
def usc_laplace(p, q, r, s, t):    
    usc = unsphericity_curvature(p, q, r, s, t) 
    lap = laplacian(r, t)
    return np.nan_to_num(usc * lap)   
"""
def usc_laplace(p, q, r, s, t):   
    from cv2 import blur 
    usc = unsphericity_curvature(p, q, r, s, t) 
    lap = laplacian(r, t)
    ul = np.nan_to_num(usc * lap) 
    ul = blur(ul, (1, 1))
    return np.clip(ul, -0.01, 0.01)


"""     
def test_usc_lap(): 
    from cv2 import blur
    from augment_by_smallsizing import smallsize_matrix_general
    matrix = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    matrix = usc_laplace(p, q, r, s, t)
    matrix = blur(matrix, (1, 1))
    matrix = np.clip(matrix, -0.01, 0.01)
    plt.imshow(matrix, cmap="Greys")
    plt.show()
"""


def test_usc_lap(): 
    from cv2 import blur
    from augment_by_smallsizing import smallsize_matrix_general
    matrix = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    matrix = usc_laplace(p, q, r, s, t)    
    plt.imshow(matrix, cmap="Greys")
    plt.show()


def test_mean_c_lap(): 
    from cv2 import blur
    from augment_by_smallsizing import smallsize_matrix_general
    matrix = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    matrix = mean_curvature(p, q, r, s, t)    
    plt.imshow(-matrix, cmap="Greys")
    plt.show()

def test_min_c_lap(): 
    from cv2 import blur
    from augment_by_smallsizing import smallsize_matrix_general
    matrix = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    matrix = minimal_curvature(p, q, r, s, t)    
    plt.imshow(-matrix, cmap="Greys")
    plt.show()

def test_t_lap(): 
    from cv2 import blur
    from augment_by_smallsizing import smallsize_matrix_general
    matrix = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    matrix = truncated_laplacian(r, t)    
    plt.imshow(matrix, cmap="Greys")
    plt.show()


def diff_eq(): 
    from cv2 import blur
    from augment_by_smallsizing import smallsize_matrix_general
    matrix = np.load("images/vieritz.npy")    
    matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    sl = slope(p, q)
    la = blur(sl, (2,2)) 
    #a = 0
    #T = np.multiply(sl, np.exp(a * la))     
    plt.imshow(la, cmap="Greys")
    plt.show()

"""
def pseudo_slope(): 
    matrix = np.load("images/vieritz.npy")    
    #matrix = smallsize_matrix_general(matrix, 2)[0] 
    p, q, r, s, t = get_gradients(matrix)
    ps = p + q 
    plt.imshow(ps, cmap="Greys")
    plt.show()
"""


def pseudo_slope(p, q):  
    # nabla operator   
    ps = p + q 
    return np.nan_to_num(ps) 



def nabla(p, q):  
    # nabla operator   
    ps = p + q 
    return np.nan_to_num(ps) 




def mhvc(p, q, r, s, t):
    #mean_horizonal_vertical_curvature
    hc = horizontal_curvature(p, q, r, s, t) 
    vc = vertical_curvature(p, q, r, s, t) 
    return 0.5 * (hc + vc) 







def shift_dataset(data_list, x, y): 
    from utils import shift
    for d in data_list: 
        for i in range(len(d)): 
            d[i] = shift(d[i], x, y) 
    return data_list



def test_shift_dataset(): 
    from file_ops import read_h5py_segmentation_dataset
    x_train, y_train, x_test, y_test = read_h5py_segmentation_dataset("compressed_segmentation_dataset.h5")
    dataset = [x_train, y_train, x_test, y_test]
    dataset = shift_dataset(dataset, 1, 1)
    x = dataset[0][0] 
    plt.imshow(x) 
    plt.show()


#test_mean_c_lap()
#test_shift_dataset()
#diff_eq()
#pseudo_slope()



# Methods for eliminating outliers in the training data

def simple_thresholding(x, denominator=15.): 
    y = []
    tmp = sorted(x) 
    m = tmp[int(len(tmp) / denominator)]
    for i in range(len(x)): 
        if x[i] < m: 
            y.append(-1) 
        else: 
            y.append(1) 
    return y    


def outlier_detection_oc_svm(data, labels, mean_type="median", classification_type="ocs"):     
    # one-class SVM on the median values of the matrix-gradients
    # data is considered to be of type list.    
    from sklearn.svm import OneClassSVM 
    n = len(data)

    median_list = []    
    for i in range(n):
        #first derivative
        p, q = get_first_gradients(data[i]) 
        data[i] = slope(p, q)
        # normalization
        data[i] += np.min(data[i]) 
        data[i] /= np.max(data[i]) 
        #square
        data[i] *= data[i] 
        # squareroot
        data[i] = np.sqrt(data[i]) 
        # get mean value
        if mean_type == "median":        
            m = np.median(data[i])
        elif mean_type == "mean": 
            m = np.mean(data[i])        
        else: 
            raise Exception("Error in outlier_detection_oc_svm: No valid mean type!")
        median_list.append(m) 
    median_list = np.asarray(median_list)
    median_list = np.reshape(median_list, (-1, 1))
    print("Comparision List: ", median_list.shape)
    # outlier detection
    if len(median_list) > 0: 
        if classification_type == "ocs": 
            clf = OneClassSVM(gamma="auto", nu=0.1).fit(median_list) 
            out = clf.predict(median_list)
        elif classification_type == "st":
            out = simple_thresholding(median_list)
        # removal of outliers from list
        for i in range(n-1, -1, -1): 
            if out[i] == -1: 
                del data[i] 
                del labels[i]
    return data, labels





def outlier_detection_lof(data, labels, mean_type="median"):     
    # Local Outlier Factor on the median values of the matrix-gradients
    # data is considered to be of type list.    
    from sklearn.neighbors import LocalOutlierFactor
    n = len(data)

    median_list = []    
    for i in range(n):
        #first derivative
        p, q = get_first_gradients(data[i]) 
        data[i] = slope(p, q)
        # normalization
        data[i] += np.min(data[i]) 
        data[i] /= np.max(data[i]) 
        #square
        data[i] *= data[i] 
        # squareroot
        data[i] = np.sqrt(data[i]) 
        # get mean value
        if mean_type == "median":        
            m = np.median(data[i])
        elif mean_type == "mean": 
            m = np.mean(data[i])
        else: 
            raise Exception("Error in outlier_detection_oc_svm: No valid mean type!")
        median_list.append(m) 
    median_list = np.asarray(median_list)
    median_list = np.reshape(median_list, (-1, 1))
    print("Median List: ", median_list.shape, median_list[0])
    # outlier detection
    clf = LocalOutlierFactor(n_neighbors=2).fit(median_list) 
    out = clf.predict(median_list)
    # removal of outliers from list
    for i in range(n-1, -1, -1): 
        if out[i] == -1: 
            del data[i] 
            del labels[i]
    return data, labels

def outlier_detection_oc_svm_without_labels(data, mean_type="median"):     
    # one-class SVM on the median values of the matrix-gradients
    # data is considered to be of type list.    
    from sklearn.svm import OneClassSVM 
    n = len(data)

    median_list = []    
    for i in range(n):
        #first derivative
        p, q = get_first_gradients(data[i]) 
        data[i] = slope(p, q)
        # normalization
        data[i] += np.min(data[i]) 
        data[i] /= np.max(data[i]) 
        #square
        data[i] *= data[i] 
        # squareroot
        data[i] = np.sqrt(data[i]) 
        # get mean value
        if mean_type == "median":        
            m = np.median(data[i])
        elif mean_type == "mean": 
            m = np.mean(data[i])
        else: 
            raise Exception("Error in outlier_detection_oc_svm: No valid mean type!")
        median_list.append(m) 
    median_list = np.asarray(median_list)
    # outlier detection
    clf = OneClassSVM(gamma="auto").fit(median_list) 
    out = clf.predict(median_list)
    # removal of outliers from list
    for i in range(n-1, -1, -1): 
        if out[i] == -1: 
            del data[i]             
    return data




def transform(  matrix, 
                method, 
                kernel_size=(41, 41), 
                direction="x", 
                azimuth=np.deg2rad(315), 
                elevation=np.deg2rad(35), 
                resolution=1, 
                compute_asvf=False, 
                compute_opns=False, 
                compute_svf=True  ): 
    # returns the transformed matrix, except for the sky view factor function, 
    # which returns a dictionary. 
    # The only mandatory arguments are the matrix to be transformed and the transformation 
    # method. The other arguments depend on the functions to which they are neccessary.  
    if not (type(matrix) is np.ndarray): 
        raise TypeError("transform: Input matrix must be 2D numpy array!")

    if method == "slrm_cv2_average": 
        return slrm_cv2_average(matrix, kernel_size) 
    elif method == "slrm_cv2_gaussian": 
        return slrm_cv2_gaussian(matrix, kernel_size) 
    elif method == "slrm_cv2_median": 
        return slrm_cv2_median(matrix, kernel_size)
    elif method == "slrm_cv2_bilateral": 
        return  slrm_cv2_bilateral(matrix, kernel_size)  
    elif method == "cv2_laplacian": 
        return  cv2_laplacian(matrix) 
    elif method == "cv2_sobel2D": 
        return  cv2_sobel2D(matrix)  
    elif method == "double_sobel": 
        return double_sobel(matrix)   
    elif method == "double_slope": 
        return  double_slope(matrix)  
    elif method == "double_slope_one_dir": 
        return  double_slope_one_dir(matrix, direction)  
    elif method == "slope": 
        p, q = get_first_gradients(matrix)
        return slope(p, q)   
    elif method == "aspect": 
        p, q = get_first_gradients(matrix)
        return  aspect(p, q)  
    elif method == "northwardness": 
        return northwardness(matrix)   
    elif method == "eastwardness": 
        return  eastwardness(matrix)  
    elif method == "plan_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return plan_curvature(p, q, r, s, t)   
    elif method == "horizontal_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return horizontal_curvature(p, q, r, s, t)   
    elif method == "vertical_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  vertical_curvature(p, q, r, s, t)  
    elif method == "difference_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  difference_curvature(p, q, r, s, t)  
    elif method == "accumulation_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  accumulation_curvature(p, q, r, s, t)  
    elif method == "ring_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  ring_curvature(p, q, r, s, t)  
    elif method == "rotor": 
        p, q, r, s, t = get_gradients(matrix)
        return rotor(p, q, r, s, t)  
    elif method == "horizontal_curvature_deflection":
        p, q, r, s, t = get_gradients(matrix)
        g, h, k, m = get_third_degree_gradients(r, t)
        return horizontal_curvature_deflection(p, q, r, s, t, g, h, k, m)
    elif method == "vertical_curvature_deflection":
        p, q, r, s, t = get_gradients(matrix)
        g, h, k, m = get_third_degree_gradients(r, t)
        return vertical_curvature_deflection(p, q, r, s, t, g, h, k, m)
    elif method == "mean_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  mean_curvature(p, q, r, s, t)  
    elif method == "gaussian_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return gaussian_curvature(p, q, r, s, t)   
    elif method == "minimal_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return minimal_curvature(p, q, r, s, t)    
    elif method == "maximal_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  maximal_curvature(p, q, r, s, t)  
    elif method == "unsphericity_curvature" or method == "usc" : 
        p, q, r, s, t = get_gradients(matrix)
        return  unsphericity_curvature(p, q, r, s, t)
    elif method == "horizontal_excess_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return  horizontal_excess_curvature(p, q, r, s, t)  
    elif method == "vertical_excess_curvature": 
        p, q, r, s, t = get_gradients(matrix)
        return vertical_excess_curvature(p, q, r, s, t)
    elif method == "laplacian":
        r, t = get_second_gradients(matrix) 
        return  laplacian(r, t)  
    elif method == "shape_index":
        p, q, r, s, t = get_gradients(matrix) 
        return   shape_index(p, q, r, s, t) 
    elif method == "curvedness":
        p, q, r, s, t = get_gradients(matrix) 
        return  curvedness(p, q, r, s, t)  
    elif method == "reflectance": 
        p, q = get_first_gradients(matrix)
        return reflectance(p, q, azimuth, elevation)   
    elif method == "insolation":
        p, q = get_first_gradients(matrix) 
        return insolation(p, q, azimuth, elevation)   
    elif method in ["pseudo_slope", "nabla"]: 
        p, q = get_first_gradients(matrix)
        return nabla(p, q) 
    elif method == "local_dominance": 
        from vis import local_dominance 
        return local_dominance(matrix) 
    elif method == "sky_illumination": 
        from vis import sky_illumination 
        return sky_illumination(matrix, resolution) 
    elif method == "svf" or method == "sky_view_factor": 
        from vis import sky_view_factor
        dict_svf = sky_view_factor( matrix, 
                                    resolution, 
                                    compute_asvf=compute_asvf, 
                                    compute_opns=compute_opns, 
                                    compute_svf=compute_svf  )
        if compute_asvf == True: 
            return dict_svf["asvf"] 
        if compute_opns == True: 
            return dict_svf["opns"] 
        else: 
            return dict_svf["svf"]
    else: 
        raise Exception("Error: No valid transformation type!")
    

def transform_dataset_list( data_list, 
                            method, 
                            kernel_size=(41, 41), 
                            direction="x", 
                            azimuth=np.deg2rad(315), 
                            elevation=np.deg2rad(35), 
                            resolution=1, 
                            compute_asvf=False, 
                            compute_opns=False, 
                            compute_svf=True, 
                            pca_components=0  ): 

    for i in data_list:   
        print()            
        for j in range(len(i)): 
            print("{} of {}".format(j, len(i))  , end="\r")
            i[j] = transform(   i[j], 
                                method, 
                                kernel_size, 
                                direction, 
                                azimuth, 
                                elevation, 
                                resolution, 
                                compute_asvf, 
                                compute_opns, 
                                compute_svf   )
            if pca_components > 0: 
                i[j] = pca(i[j], pca_components)
    return data_list


def save_matrix_as_transformed_png(filename, matrix, methods):
    #filename: folder and name where to save the image.
    # methods: list of one OR three transformation methods.     
    from utils import save_rgb_image_from_numpy_matrices
    if len(methods) == 3: 
        c1 = transform(matrix, methods[0]) 
        c2 = transform(matrix, methods[1]) 
        c3 = transform(matrix, methods[2]) 
    else: 
        c1 = transform(matrix, methods[0]) 
        c2 = c1.copy() 
        c3 = c1.copy() 
    save_rgb_image_from_numpy_matrices(filename, [c1, c2, c3]) 


def check_line_for_bboxes(line):
    # checks if there is incomplete bbox data in a csv-line.
    s = line.split(",") 
    #print(s[-4])
    try: 
        xmin = int(s[-4]) 
        ymin = int(s[-3]) 
        xmax = int(s[-2]) 
        ymax = int(s[-1]) 
        if max(xmin,ymin,xmax,ymax) > 63:
            print(line)
    except:
        print(line) 



def merge_csv_files(path): 
    import glob 
    full_data = ["filename,width,height,class,xmin,ymin,xmax,ymax"]

    for filename in glob.glob(path + "/*.csv"): 
        print(filename)
        f = open(filename) 
        d = f.readlines() 
        f.close()  
        print(d[-1])      
        d = d[1:]

        full_data = full_data + d 
    for i in range(len(full_data)): 
        full_data[i] = full_data[i].strip("\n")
    
    #d_splitted = full_data.split("\n") 
    while '' in full_data: 
        full_data.remove('')
    for i in range(len(full_data)): 
        if full_data[i].count(",") < 7: 
            print("Bad line in full_data: ", [full_data[i]])
    full_data = "\n".join(full_data)
    s = open(path + "/full_csv_data.csv", "w") 
    s.write(full_data) 
    s.close() 
    



def save_all_matrices_as_png(path, methods):     
    import pandas as pd 
    train_labels = [] 
    test_labels = []
    s0 = open(path + "/full_csv_data.csv") 
    d = s0.readlines() 
    s0.close()
    df = pd.read_csv(path + "/full_csv_data.csv")    
    print(df["filename"][0])
    s = set(df["filename"])
    s = list(s) 
    #print(len(s))
    for filename in s: 
        for line in d: 
            if filename in line: 
                if "/train/" in line: 
                    train_labels.append(line)
                elif "/test/" in line: 
                    test_labels.append(line) 
        f_npy = filename.replace(".png", ".npy") 
        f_png = filename.replace("npy_matrices", "images") 
        matrix = np.load(f_npy) 
        save_matrix_as_transformed_png(f_png, matrix, methods)

    train_labels_s = "filename,width,height,class,xmin,ymin,xmax,ymax\n"
    test_labels_s = "filename,width,height,class,xmin,ymin,xmax,ymax\n"
    index = 0
    for line in train_labels:
        print(index, " of ", len(train_labels))         
        train_labels_s = train_labels_s + line  
        index += 1
        #if index == 10: 
        #    break
    index = 0
    for line in test_labels: 
        print(index, " of ", len(test_labels), end="\r") 
        test_labels_s = test_labels_s + line 
        index += 1
        #if index == 10: 
        #    break
    w = open(path + "/train_labels.csv", "w") 
    w.write(train_labels_s) 
    w.close()  
    w = open(path + "/test_labels.csv", "w") 
    w.write(test_labels_s) 
    w.close() 
    


def save_all_matrices_as_png_reduced(path, methods):     
    import pandas as pd 
    train_labels = [] 
    test_labels = []
    s0 = open(path + "/full_csv_data.csv") 
    d = s0.readlines() 
    s0.close()
    df = pd.read_csv(path + "/full_csv_data.csv")    
    print(df["filename"][0])
    s = set(df["filename"])
    s = list(s) 
    #print(len(s))
    reduction_factor = 2
    index = 0 
    #for filename in s: 
    for index in range(0, len(s), reduction_factor):
        filename = s[index]
        #if index % reduction_factor == 0:
        for line in d: 
            if filename in line: 
                if "/train/" in line: 
                    train_labels.append(line)
                elif "/test/" in line: 
                    test_labels.append(line) 
        f_npy = filename.replace(".png", ".npy") 
        f_png = filename.replace("npy_matrices", "images") 
        matrix = np.load(f_npy) 
        save_matrix_as_transformed_png(f_png, matrix, methods)

    train_labels_s = "filename,width,height,class,xmin,ymin,xmax,ymax\n"
    test_labels_s = "filename,width,height,class,xmin,ymin,xmax,ymax\n"
    index = 0
    for line in train_labels:
        print(index, " of ", len(train_labels), end="\r")         
        train_labels_s = train_labels_s + line  
        index += 1
        #if index == 10: 
        #    break
    index = 0
    for line in test_labels: 
        print(index, " of ", len(test_labels), end="\r") 
        test_labels_s = test_labels_s + line 
        index += 1
        #if index == 10: 
        #    break
    w = open(path + "/train_labels_reduced_" + str(reduction_factor) + ".csv", "w") 
    w.write(train_labels_s) 
    w.close()  
    w = open(path + "/test_labels_reduced_" + str(reduction_factor) + ".csv", "w") 
    w.write(test_labels_s) 
    w.close() 


def show_png_with_boxes(path):     
    import pandas as pd 
    from utils import show_image_with_bboxes   #(image, bbox_list)
    import sys
    from bounding_box import BBox
    from cv2 import imread
    s0 = open(path + "/train_labels.csv") 
    d = s0.readlines() 
    s0.close()
    df = pd.read_csv(path + "/train_labels.csv")    
    print(df["filename"][0])
    s = set(df["filename"])
    s = list(s) 
    reduction_factor = 10
    index = 0 
    #for filename in s: 
    for index in range(0, len(s), reduction_factor):
        filename = s[index]
        img = imread(path + "/" + filename)
        #print(img.shape)
        bbox_list = []
        for line in d:             
            if filename in line: 
                splitted_line = line.split(",") 
                xmin = int(splitted_line[-4])
                ymin = int(splitted_line[-3])
                xmax = int(splitted_line[-2])
                ymax = int(splitted_line[-1])
                bbox = BBox(xmin, xmax, ymin, ymax, filename, "gravemound", index) #, is_voc=True
                bbox_list.append(bbox)
                #images/train/NVE_Driva_20161_75_59.png
        show_image_with_bboxes(img, bbox_list)

                
def check_csv_lines(filename, outfile_name): 
    csv_file = open(filename) 
    lines = csv_file.readlines()
    csv_file.close()
    #for line in lines[158380:158383]: 
    #    print(line)
    new = []
    for line in lines: 
        line = line.replace("\n", "")
        if line.count(",") == 7: 
            new.append(line) 
        elif line.count(",") == 14: 
            n = line.split("images") 
            new.append("images" + n[0]) 
            new.append("images" + n[1]) 
        else: 
            print(line.count(","), "line: ", line) 
    outfile = open(outfile_name, "w")
    for i in range(len(new)): 
        new[i] = new[i].replace("images/rcnn/npy_matrices/train/", "images/train/")
        new[i] = new[i].replace("images/rcnn/npy_matrices/test/", "images/test/")
        if i == 0: 
            if new[0].count(",") != 7: 
                print(new[i].count(","), "line: ", new[i])
            outfile.write(new[0])
        else: 
            if new[i].count(",") != 7: 
                print(new[i].count(","), "line: ", new[i])
            outfile.write("\n" + new[i])

        
def check_csv_line(filename): 
    csv_file = open(filename) 
    lines = csv_file.readlines()
    csv_file.close()
    break_dict = {}
    for line in lines: 
        if line.count("\n") in break_dict: 
            break_dict[line.count("\n")] += 1 
        else: 
            break_dict[line.count("\n")] = 1
        #print(line)
    print(break_dict)


def check_csv_bboxes(filename): 
    csv_file = open(filename) 
    lines = csv_file.readlines()
    csv_file.close()    
    for line in lines: 
        if line != lines[0]: 
            check_line_for_bboxes(line)
            
        
