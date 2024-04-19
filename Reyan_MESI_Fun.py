import numpy as np

def mesi_fun(p, param):
    """
    Evaluates the MESI equation.
    :param p: array_like, shape (3,) or (4,)
        Parameters.
        If length(p) == 4:
            p[0] = beta
            p[1] = rho
            p[2] = tau_c
            p[3] = nu
        If length(p) == 3:
            p[0] = rho
            p[1] = tau_c
            p[2] = nu
    :param param: dict
        Dictionary containing additional parameters.
        param['T']: array_like, shape (n,)
            Vector of exposure times (in seconds) used during MESI.
        param['beta']: array_like, shape (N,) or None
            Matrix of Beta values for use during fitting organized as MxN.
            None if beta is not fixed.
    :return: array_like, shape (n,)
        MESI equation evaluated for a given set of inputs and exposure times.
    """

    T = param['T']

    # Prepend fixed beta onto input list
    if 'beta' in param:
        p = np.concatenate(([param['beta']], p))

    F = (
        p[0] * p[1]**2 * p[2]**2 * (np.exp(-2 * T / p[2]) + 2 * T / p[2] - 1) / (2 * T**2) +
        4 * p[0] * p[1] * (1 - p[1]) * p[2]**2 * (np.exp(-T / p[2]) + T / p[2] - 1) / (T**2) +
        p[0] * (1 - p[1])**2 + p[3]
    )

    return F
