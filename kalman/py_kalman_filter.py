import numpy as np

def py_kalman_filter(y,  # T+1xn    (data: endogenous, observed)
                     H,  # nxkxT+1  (parameters)
                     mu, # kx1      (parameters)
                     F,  # kxk      (parameters)
                     R,  # kxk      (parameters: covariance matrix)
                     Q,  # kxk      (parameters: covariance matrix)
                     z=None,  # T+1xr    (data: weakly exogenous, observed)
                     A=None,  # nxr      (parameters)
                     beta_tt_init=None,
                     P_tt_init=None):

    T = y.shape[0]
    n = y.shape[1]
    k = mu.shape[0]
    time_varying_H = H.shape[-1] == T+1

    # Check if we have an exog matrix
    if z is not None and A is not None:
        r = z.shape[1]
    else:
        r = 0
    
    # Allocate memory for variables
    beta_tt = np.zeros((T+1,k))
    P_tt = np.zeros((T+1,k,k))
    beta_tt1 = np.zeros((T+1,k))
    P_tt1 = np.zeros((T+1,k,k))
    y_tt1 = np.zeros((T+1,n))
    eta_tt1 = np.zeros((T+1,n))
    f_tt1 = np.zeros((T+1,n,n))
    gain = np.zeros((T+1,k,n))
    ll = np.zeros((T+1,))
    
    # Initial values
    if beta_tt_init is None:
        beta_tt[0] = np.linalg.inv(np.eye(k) - F).dot(mu)
    else:
        beta_tt[0] = beta_tt_init
    if P_tt_init is None:
        P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k)
    else:
        P_tt[0] = P_tt_init

    # Iterate forwards
    H_view = H[:,:,0]
    for t in range(1,T+1):
        if time_varying_H:
            H_view = H[:,:,t]
        
        # Prediction
        beta_tt1[t] = mu + F.dot(beta_tt[t-1])
        P_tt1[t]    = F.dot(P_tt[t-1]).dot(F.T) + Q
        y_tt1[t]    = H_view.dot(beta_tt1[t])
        if r > 0:
            y_tt1[t] += A.dot(z[t-1])
        eta_tt1[t]  = y[t-1] - y_tt1[t]
        PHT = P_tt1[t].dot(H_view.T)
        f_tt1[t]    = H_view.dot(PHT) + R
        if n == 1:
            f_inv = 1/f_tt1[t]
            det = np.abs(f_tt1[t])
        else:
            f_inv = np.linalg.inv(f_tt1[t])
            det = np.linalg.det(f_tt1[t])

        # Log-likelihood as byproduct
        ll[t] = -0.5*np.log(2*np.pi*det) - 0.5*eta_tt1[t].T.dot(f_inv).dot(eta_tt1[t])
        
        # Updating
        gain[t] = PHT.dot(f_inv)
        beta_tt[t] = beta_tt1[t] + gain[t].dot(eta_tt1[t])
        P_tt[t] = P_tt1[t] - gain[t].dot(H_view).dot(P_tt1[t])
    
    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, gain, ll
