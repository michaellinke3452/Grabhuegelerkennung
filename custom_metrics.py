import tensorflow as tf
from utils import remove_nan
import sklearn.metrics as sm

def f1(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1]) 
    n = tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))    
    a = tf.reduce_sum( tf.math.multiply(y_true_f, y_true_f))
    b = tf.math.reduce_sum( tf.math.multiply(y_pred_f, y_pred_f))
    return (2. * n + smooth) / (a + b + smooth)


def f1_loss(y_true, y_pred):
    return 1. - f1(y_true, y_pred)


def threshold(y_true, y_pred): 
    # use positive_MSE instead!
    from tensorflow.math import reduce_sum, multiply
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1]) 
    intersection = multiply(y_true_f, y_pred_f)
    true_squared = multiply(y_true_f, y_true_f)    

    return tf.keras.losses.MSE(true_squared, intersection)


def positive_MSE(y_true, y_pred): 
    from tensorflow.math import reduce_sum, multiply
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1]) 
    intersection = multiply(y_true_f, y_pred_f)
    true_squared = multiply(y_true_f, y_true_f)    

    return tf.keras.losses.MSE(true_squared, intersection)



def positive_MAE(y_true, y_pred): 
    from tensorflow.math import reduce_sum, multiply
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1]) 
    intersection = multiply(y_true_f, y_pred_f)
    true_squared = multiply(y_true_f, y_true_f)    

    return tf.keras.losses.MAE(true_squared, intersection)


def sklearn_precision(y_true, y_pred): 
    return sm.precision_score(y_true, y_pred)

def sklearn_recall(y_true, y_pred): 
    return sm.recall_score(y_true, y_pred) 

def sklearn_balanced_accuracy(y_true, y_pred): 
    return sm.balanced_accuracy_score(y_true, y_pred) 

def sklearn_ba_loss(y_true, y_pred): 
    return 1. - sklearn_balanced_accuracy(y_true, y_pred) 

def sklearn_f1_score(y_true, y_pred): 
    return sm.f1_score(y_true, y_pred) 

def sklearn_f1_loss(y_true, y_pred): 
    return 1. - sklearn_f1_score(y_true, y_pred) 

def sklearn_auc(y_true, y_pred): 
    return sm.auc(y_true, y_pred) 

def sklearn_auc_loss(y_true, y_pred): 
    return 1. - sklearn_auc(y_true, y_pred)

def sklearn_fbeta_score(y_true, y_pred): 
    return sm.fbeta_score(y_true, y_pred) 

def sklearn_fbeta_loss(y_true, y_pred): 
    return 1. - sklearn_fbeta_score(y_true, y_pred)

def sklearn_mcc(y_true, y_pred): 
    return sm.matthews_corrcoef(y_true, y_pred) 

def sklearn_mcc_loss(y_true, y_pred): 
    return 1. - sklearn_mcc(y_true, y_pred)


def true_positives(y_true, y_pred): 
    from tensorflow.math import reduce_sum, multiply
    true = tf.reshape(y_true, [-1]) 
    pred = tf.reshape(y_pred, [-1]) 
    return reduce_sum(multiply(true, pred))


def false_negatives(y_true, y_pred): 
    from tensorflow.math import reduce_sum, multiply
    true = tf.reshape(y_true, [-1]) 
    pred = tf.reshape(y_pred, [-1]) 
    return reduce_sum(multiply(true, (1 - pred)))

def false_positives(y_true, y_pred): 
    from tensorflow.math import reduce_sum, multiply
    true = tf.reshape(y_true, [-1]) 
    pred = tf.reshape(y_pred, [-1]) 
    return reduce_sum(multiply((1 - true), pred))


def true_negatives(y_true, y_pred): 
    from tensorflow.math import reduce_sum, multiply
    true = tf.reshape(y_true, [-1]) 
    pred = tf.reshape(y_pred, [-1]) 
    return reduce_sum(multiply((1 - true), (1 - pred)))


def get_basic_metrics(y_true, y_pred):     
    # Returns: A dictionary of true positives (TP), true negatives (TN), 
    #          false positives (FP) and false negatives (FN). 
    # Alternative to: evaluate_confusion_matrix(y_true, y_pred)   
    import tensorflow.keras.backend as K
    values = {} 
    true = K.flatten(y_true) 
    pred = K.flatten(y_pred)     
    values["TP"] = K.sum(true * pred)
    values["FN"] = K.sum(true * (1 - pred)) 
    values["FP"] = K.sum((1 - true) * pred) 
    values["TN"] = K.sum((1 - true) * (1 - pred)) 
    return values


def get_confusion_matrix(y_true, y_pred):     
    #true = tf.round(tf.reshape(y_true, [-1]))  
    #pred = tf.round(tf.reshape(y_pred, [-1]))       #, tf.int32
    true = tf.reshape(y_true, [-1])  
    pred = tf.reshape(y_pred, [-1])  
    return tf.cast(tf.confusion_matrix(true, pred, num_classes=2), tf.float32)


def evaluate_confusion_matrix(y_true, y_pred): 
    # creates a confusion matrix based on ground-truth (y_true) and 
    # the outcome of a prediction (y_pred). 
    # Returns: A dictionary of true positives (TP), true negatives (TN), 
    #          false positives (FP) and false negatives (FN).
    # Alternative to: get_basic_metrics(y_true, y_pred)
    cm = get_confusion_matrix(y_true, y_pred) 
    values = {}
    values["TP"] = cm[1][1] 
    values["TN"] = cm[0][0] 
    values["FP"] = cm[0][1] 
    values["FN"] = cm[1][0]
    return values 


def precision(y_true, y_pred, smooth=1):
    # computes the precision, also called positive predictive value (PPV)
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1]) 
    TP = tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))
    FP = tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, y_pred_f))    
    return (TP + smooth) / (TP + FP + smooth)


def recall(y_true, y_pred, smooth=1):
    # computes the recall, also called sensitivity, hit rate, or true positive rate (TPR)
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1]) 
    TP = tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))
    FN = tf.math.reduce_sum( tf.math.multiply(y_true_f, 1 - y_pred_f))    
    return (TP + smooth) / (TP + FN + smooth)


def f1_2(y_true, y_pred, smooth=1): 
    a = precision(y_true, y_pred, smooth) * recall(y_true, y_pred, smooth) 
    b = precision(y_true, y_pred, smooth) + recall(y_true, y_pred, smooth) 
    return 2 * (a / b)


def f1_2_loss(y_true, y_pred, smooth=1):     
    return 1 - f1_2(y_true, y_pred, smooth)


def f_beta(y_true, y_pred, beta=0.3, smooth=1): 
    # computes the F-Score. If beta == 1, f_beta is aequivalent to f1-score.
    p = precision(y_true, y_pred, smooth)
    r = recall(y_true, y_pred, smooth)
    a = (1 + beta**2) * p * r
    b = (beta**2) * p + r 
    return a / b

def f_beta_loss(y_true, y_pred): 
    # computes a loss function based on the f-score. Beta is hardcoded as 0.2.
    return 1 - f_beta(y_true, y_pred, beta=0.2, smooth=1)



def fallout(y_true, y_pred): 
    # computes the fall-out, also called false positive rate (FPR)    
    FP = false_positives(y_true, y_pred)
    TN = true_negatives(y_true, y_pred)
    return FP / (FP + TN)    


def missrate(y_true, y_pred): 
    # computes the miss-rate, also called false negative rate (FNR)    
    FN = false_negatives(y_true, y_pred)
    TP = true_positives(y_true, y_pred)
    return FN / (FN + TP)    


def dsc(y_true, y_pred, smooth=1.): 
    import tensorflow.keras.backend as K
    true = K.flatten(y_true)
    pred = K.flatten(y_pred)
    TP = K.sum(true * pred)
    FP = K.sum((1. - true) * pred)
    FN = K.sum(true * (1. - pred))        
    a = 2. * TP + smooth 
    b = 2. * TP + FP + FN + smooth 
    return a / b

def dsc_loss(y_true, y_pred): 
    return 1. - dsc(y_true, y_pred)


def fowlkes_mallows_index(y_true, y_pred):
    # computes the geometric mean of recall and precision 
    #values = evaluate_confusion_matrix(y_true, y_pred)
    return tf.math.sqrt(precision(y_true, y_pred) * recall(y_true, y_pred)) 



def mcc(y_true, y_pred, smooth=1.): 
    # computes the matthews_correlation_coefficient.
    # it includes normalization, but in a wrong way. For correct normalization,
    # choose mcc_normalized.    
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1])
    
    TP = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))), 0.)
    FP = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, y_pred_f))), 0.)
    FN = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(y_true_f, 1 - y_pred_f))), 0.)
    TN = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, 1 - y_pred_f))), 0.)
    
    tpfp = (TP + FP) 
    tpfn = (TP + FN) 
    tnfp = (TN + FP) 
    tnfn = (TN + FN) 
    
    a = TP * TN - FP * FN + smooth
    b = tpfp * tpfn * tnfp * tnfn + smooth
    b = tf.math.abs(b)
    b = tf.math.sqrt(b) 
    return tf.math.abs((a + 1) / b) / 2 


def mcc_normalized(y_true, y_pred, smooth=1.): 
    # computes the normalized matthews_correlation_coefficient    
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1])
    
    TP = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))), 0.)
    FP = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, y_pred_f))), 0.)
    FN = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(y_true_f, 1 - y_pred_f))), 0.)
    TN = remove_nan(tf.math.abs(tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, 1 - y_pred_f))), 0.)
    
    tpfp = (TP + FP) 
    tpfn = (TP + FN) 
    tnfp = (TN + FP) 
    tnfn = (TN + FN) 
    
    a = TP * TN - FP * FN + smooth
    b = tpfp * tpfn * tnfp * tnfn + smooth
    b = tf.math.abs(b)
    b = tf.math.sqrt(b) 
    return (tf.math.abs(a / b) + 1)/ 2 


def mcc_loss(y_true, y_pred):     
    return remove_nan(1. - mcc(y_true, y_pred), 0.00000001) 


def mcc_normalized_loss(y_true, y_pred):     
    return remove_nan(1. - mcc(y_true, y_pred), 0.00000001) 


def dice_by_cm(y_true, y_pred): 
    cm = get_confusion_matrix(y_true, y_pred)
    TP = cm[1][1] 
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0] 
    a = 2. * TP + tf.constant(1.)
    b = 2. * TP + FP + FN + tf.constant(1.)
    return a / b 


def dice_loss_by_cm(y_true, y_pred): 
    return tf.constant(1.) - dice_by_cm(y_true, y_pred) 


def focal_tversky_loss(y_true, y_pred, a=0.5, b=0.5, gamma=1.): 
    # computes the Focal Tversky Loss. 
    # gamma > 1 causes a higher loss gradient for TI-outcomes < 0.5 and 
    # thus makes the model focus on more difficult cases. 
    #import tensorflow.keras.backend as K    
    return tf.math.pow((1. - tversky_index(y_true, y_pred, a, b)), gamma) 


def tversky_index(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1):    
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1])
    
    TP = tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))
    FP = tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, y_pred_f))
    FN = tf.math.reduce_sum( tf.math.multiply(y_true_f, 1 - y_pred_f))
    n = TP + smooth
    d = TP + (alpha * FN) + (beta * FP) + smooth
    return n / d


def tversky_loss(y_true, y_pred):
    #import tensorflow.keras.backend as K
    alpha = 0.3 
    beta = 0.7
    return 1 - tversky_index(y_true,y_pred, alpha=alpha, beta=beta)


def ftl(y_true, y_pred, g=0.75):
    # computes the Focal Tversky Loss.
    import tensorflow.keras.backend as K    
    return  K.pow((1 - tversky_index(y_true,y_pred)), g)


def weighed_tversky_index(y_true, y_pred, smooth=1): 
    y_true_f = tf.reshape(y_true, [-1]) 
    y_pred_f = tf.reshape(y_pred, [-1])
    N = tf.shape(y_true_f)[0]
    N = tf.to_float(N) 
    T = tf.math.reduce_sum(y_true_f) 
    alpha = T / N 
    TP = tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))
    FP = tf.math.reduce_sum( tf.math.multiply(1 - y_true_f, y_pred_f))
    FN = tf.math.reduce_sum( tf.math.multiply(y_true_f, 1 - y_pred_f))
    n = TP + smooth
    d = TP + (alpha * FN) + ((1 - alpha) * FP) + smooth
    return n / d


def weighed_tversky_loss(y_true, y_pred, smooth=1): 
    return 1 - weighed_tversky_index(y_true, y_pred, smooth=smooth)




# experimental


def weighed_f1(y_true, y_pred, g=0.7, h=0.3, smooth=1):
    y_true_f = tf.reshape(y_true, [-1, 1]) 
    y_pred_f = tf.reshape(y_pred, [-1, 1]) 
    n = tf.math.reduce_sum( tf.math.multiply(y_true_f, y_pred_f))
    #return (2. * n + smooth) / (tf.reduce_sum( tf.math.multiply(y_true_f, y_true_f)) + tf.math.reduce_sum( tf.math.multiply(y_pred_f, y_pred_f)) + smooth)
    a = tf.reduce_sum( tf.math.multiply(y_true_f, y_true_f))
    b = tf.math.reduce_sum( tf.math.multiply(y_pred_f, y_pred_f))
    return (2. * n + smooth) / (g * a + h * b + smooth)

def weighed_f1_loss(y_true, y_pred): 
    return 1. - weighed_f1(y_true, y_pred)



def precision_loss(y_true, y_pred): 
    return 1. - precision(y_true, y_pred)


def fowlkes_mallows_loss(y_true, y_pred): 
    return 1. - fowlkes_mallows_index(y_true, y_pred)


def precision_recall_mean(y_true, y_pred): 
    return 1. - ((precision(y_true, y_pred) + recall(y_true, y_pred)) / 2. )


def fallout_missrate(y_true, y_pred, smooth=1): 
    return (fallout(y_true, y_pred) * missrate(y_true, y_pred) + smooth) / (fallout(y_true, y_pred) + missrate(y_true, y_pred) + smooth) 