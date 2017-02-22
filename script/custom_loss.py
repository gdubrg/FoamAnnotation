from keras.backend import theano_backend as K
from theano import tensor as T
import theano
import cv2
import numpy as np
import scipy.stats as st

def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))
