import numpy as np
from scipy.optimize import least_squares
from scipy.io import loadmat 
import csv



def fitMESI_TR(T, Ksq, Beta=None):
    DEBUG = False  # Flag to enable verbose output while debugging

    # Check that data and exposure times have the same dimensions

    

    n = len(T)
    A = Ksq.shape
    if n != A[-1]:
        raise ValueError(f"Mismatch between number of exposure times ({n}) and number of frames in data ({A[-1]})")

    # If defined, check that beta and data have the same dimensions
    fixedBeta = False
    if Beta is not None:  # means that beta is fixed
        fixedBeta = True
        if Beta.shape != A[:-1]:
            raise ValueError(f"Shape of BETA matrix {Beta.shape} does not match shape of data {A[:-1]}")

    # Restructure data for processing
    N = np.prod(A[:-1])
    Ksq = np.transpose(Ksq, axes=np.roll(np.arange(Ksq.ndim), -1)).reshape(N, n)

    # Preallocate storage
    fitted = np.zeros((N, 4))
    R2 = np.zeros(N)

    # Fitting parameters
    OPTS = {'maxiter': 1000}
    param = {'T': T}

    if fixedBeta:
        fitted[:, 0] = Beta.reshape(N)
        LB = [np.finfo(float).eps, np.finfo(float).eps, np.finfo(float).eps]
        UB = [1, 1, 1]
        if DEBUG:
            print(f"Performing Trust-Region SI fitting with fixed BETA on {N} datapoints")
    else:
        LB = [np.finfo(float).eps] * 4
        UB = [1] * 4
        if DEBUG:
            print(f"Performing Trust-Region MESI fitting on {N} datapoints")

    if DEBUG:
        import time
        start_time = time.time()

    for i in range(N):
        Y = Ksq[i, :]

        # Perform MESI fit with fixed Beta
        if fixedBeta:
            P0 = [1, 1e-3, 1e-2]
            param['beta'] = Beta[i]
            fitted[i, 1:] = least_squares(mesi_optimize, P0, bounds=(LB, UB), method='trf', options=OPTS,
                                          args=(param, Y)).x
        else:  # Perform full MESI fit on all variables
            P0 = [Y[0], 1, 1e-3, 1e-2]
            fitted[i, :] = least_squares(mesi_optimize, P0, bounds=(LB, UB), method='trf', options=OPTS,
                                         args=(param, Y)).x

        R2[i] = 1 - np.sum((Y - mesi_fun(fitted[i], param)) ** 2) / np.sum((Y - np.mean(Y)) ** 2)

    if DEBUG:
        print(f"Total Processing Time: {time.time() - start_time:.4f}s ({(time.time() - start_time) / N:.4f}s per fit)")

    # Reshape data back to original dimensions
    fitted = fitted.reshape(A[:-1] + (4,))
    if len(A) > 2:
        R2 = R2.reshape(A[:-1])
    return fitted, R2


def mesi_optimize(params, param, Y):
    beta, rho, tau_c, nu = params
    T = param['T']
    Y_fit = mesi_fun(params, param)
    return (Y - Y_fit) / Y_fit


def mesi_fun(params, param):
    beta, rho, tau_c, nu = params
    T = param['T']
    return beta * (1 + (T / tau_c) ** (2 * nu)) ** (-rho / nu)

x = []

with open('Cropped_Data_Labels.csv', 'r', encoding="latin-1") as file1:
    with open('Cropped_Data.csv', 'r', encoding="latin-1") as file2:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)

        for row1, row2 in zip(reader1, reader2):
            x.append((row1, row2))

print(x)
fitMESI_TR(x, )

# Example usage:
# fitted, R2 = fitMESI_TR(T, Ksq)
