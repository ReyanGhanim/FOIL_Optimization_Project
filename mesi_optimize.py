import numpy as np


def mesi_optimize(p, param, Y):
    """
    Optimization function for use with MESI lsqnonlin() fitting
    F = mesi_optimize(p, param, Y) computes the difference between experimental
    data (Y) and the MESI equation evaluated for a given set of inputs (p)
    and exposure times (param['T']) for use with lsqnonlin() fitting. Reducing
    the number of inputs to 3 and defining param['beta'] evaluates the
    MESI equation with a fixed beta value.

    [F,J] = mesi_optimize(p, param, Y) also evaluates the Jacobian of the
    MESI equation for the given inputs and exposure times.

    If length(p) == 4           If length(p) == 3
      p[0] = beta                 p[0] = rho
      p[1] = rho                  p[1] = tau_c
      p[2] = tau_c                p[2] = n
      p[3] = n

    param
     |- T                        |- T
                                  |- beta
    """
    # Compute the model prediction
    F = mesi_fun(p, param) - Y

    # Check if the Jacobian is also requested
    J = None
    if nargout > 1:
        J = mesi_jac(p, param)

    # Return the function value and the Jacobian if requested
    if J is not None:
        return F, J
    else:
        return F
