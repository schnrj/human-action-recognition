import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    mat = scipy.io.loadmat('dataset/Inertial/Inertial/a1_s1_t1_inertial.mat')
    data = mat['d_iner']  # Correct key from your file
    return data.astype(np.float32)
