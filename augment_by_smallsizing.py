import numpy as np 
import matplotlib.pyplot as plt


def smallsize_matrix(m): 
    tmp = [] 
    for i in range(0, len(m), 2): 
        tmp.append(m[i]) 
    m2 = []
    for t in tmp:
        d = [t[i] for i in range(0, len(t), 2)] 
        m2.append(d) 
    m2 = np.asanyarray(m2) 
    return m2   


def smallsize_matrix2(m): 
    tmp = [] 
    for i in range(0, len(m), 2): 
        tmp.append(m[i]) 
    m2 = []
    for t in tmp:
        d = [t[i] for i in range(1, len(t), 2)] 
        m2.append(d) 
    m2 = np.asanyarray(m2) 
    return m2   



def smallsize_matrix3(m): 
    tmp = [] 
    for i in range(1, len(m), 2): 
        tmp.append(m[i]) 
    m2 = []
    for t in tmp:
        d = [t[i] for i in range(0, len(t), 2)] 
        m2.append(d) 
    m2 = np.asanyarray(m2) 
    return m2   


def smallsize_matrix4(m): 
    tmp = [] 
    for i in range(1, len(m), 2): 
        tmp.append(m[i]) 
    m2 = []
    for t in tmp:
        d = [t[i] for i in range(1, len(t), 2)] 
        m2.append(d) 
    m2 = np.asanyarray(m2) 
    return m2   


def get_smallsized_matrix(m, x, y, steps): 
    tmp = [] 
    for i in range(x, len(m), steps): 
        tmp.append(m[i]) 
    m2 = []
    for t in tmp:
        d = [t[i] for i in range(y, len(t), steps)] 
        m2.append(d) 
    m2 = np.asanyarray(m2) 
    return m2   


def smallsize_matrix_general(m, steps): 
    smallsized_matrices = []    
    for x in range(steps):
        for y in range(steps): 
            matrix = get_smallsized_matrix(m, x, y, steps)
            smallsized_matrices.append(matrix)
    return smallsized_matrices


def smallsize_test():
    rgmd = np.load("raw_gm_data.npy") 
    print(rgmd[0][0])
    plt.imshow(rgmd[0][0])
    plt.show()
    smallsized_matrices = smallsize_matrix_general(rgmd[0][0], 3)
    print(len(smallsized_matrices))
    plt.imshow(smallsized_matrices[0])
    plt.show()


def get_smallsized_matrices(filename): 
    matrix = np.load(filename)
    smallsized_matrices = smallsize_matrix_general(matrix, 8)
    print("Number of Matrices: ", len(smallsized_matrices))
    
    for i in range(len(smallsized_matrices)):
        plt.imshow(smallsized_matrices[i])
        plt.show()
    
    return smallsized_matrices


def get_unreal_line(n):
    m = np.zeros((100, 100), dtype=np.int32)    
    for i in range(100): 
        for j in range(100): 
            for k in range(1, n):
                if i % k == 0 or j % k == 0: 
                    m[i][j] = k 
    return m

def get_test_matrix(n, k):
    kernel = np.arange((n)**2) 
    kernel = kernel.reshape((n,n)) 
    for i in range(k):
        kernel2 = np.concatenate((kernel, kernel))
        kernel3 = np.concatenate((kernel2, kernel2), axis=1)
        kernel = kernel3
    return kernel



def test_smallsizing(): 
    m = get_test_matrix(8, 2)
    smallsized = smallsize_matrix_general(m, 8)
    for s in smallsized: 
        plt.imshow(s)
        plt.show()


#oppdal = "numpy_matrices/Oppdal_12pkt_2011.npy"
#get_smallsized_matrices(oppdal)
#test_smallsizing()