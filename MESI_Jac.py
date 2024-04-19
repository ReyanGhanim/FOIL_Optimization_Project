import numpy as np

def mesi_jac(p, param):
    """
    Evaluates the Jacobian of the MESI equation.
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
    :return: array_like, shape (n, 3) or (n, 4)
        Jacobian of the MESI equation for the given inputs and exposure times.
    """
    T = param['T']

    # Prepend fixed beta onto input list
    if 'beta' in param:
        p = np.concatenate(([param['beta']], p))

    A = np.exp(-2 * T / p[2]) + 2 * T / p[2] - 1
    B = np.exp(-T / p[2]) + T / p[2] - 1
    C = 1 - p[1]

    J = np.ones((len(T), 4))

    J[:, 1] = (
        p[0] * p[1] * p[2]**2 * A / T**2 - 4 * p[0] * p[1] * p[2]**2 * B / T**2 +
        4 * p[0] * p[2]**2 * C * B / T**2 - 2 * p[0] * C
    )

    J[:, 2] = (
        p[0] * p[1]**2 * (np.exp(-2 * T / p[2]) - 1) / T +
        p[0] * p[1]**2 * p[2] * A / T**2 +
        4 * p[0] * p[1] * C * (np.exp(-T / p[2]) - 1) / T +
        8 * p[0] * p[1] * p[2] * C * B / T**2
    )

    if 'beta' in param:  # Drop first column if fixed beta MESI fit
        J = np.delete(J, 0, axis=1)
    else:  # Otherwise evaluate normally
        J[:, 0] = (
            p[1]**2 * p[2]**2 * A / (2 * T**2) +
            4 * p[1] * p[2]**2 * C * B / T**2 +
            C**2
        )

    return J
