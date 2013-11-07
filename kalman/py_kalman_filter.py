import numpy as np

def py_kalman_filter(y,  # T+1xn    (data: endogenous, observed)
                     z,  # T+1xr    (data: weakly exogenous, observed)
                     A,  # nxr      (parameters)
                     H,  # nxkxT+1  (parameters)
                     mu, # kx1      (parameters)
                     F,  # kxk      (parameters)
                     R,  # kxk      (parameters: covariance matrix)
                     Q,  # kxk      (parameters: covariance matrix)
                     beta_tt_init=None,
                     P_tt_init=None):

    T = y.shape[0]-1
    n = y.shape[1]
    r = z.shape[1]
    k = mu.shape[0]
    time_varying_H = H.shape[-1] == T+1
    
    # Allocate memory for variables
    beta_tt = np.zeros((T+1,k))  # T+1xkx1
    P_tt = np.zeros((T+1,k,k))     # T+1xkxk
    beta_tt1 = np.zeros((T+1,k)) # T+1xkx1
    P_tt1 = np.zeros((T+1,k,k))    # T+1xkxk
    y_tt1 = np.zeros((T+1,n))    # T+1xnx1
    eta_tt1 = np.zeros((T+1,n))  # T+1xnx1
    f_tt1 = np.zeros((T+1,n,n))    # T+1xnxn
    gain = np.zeros((T+1,k,n))     # T+1xkxn
    ll = np.zeros((T+1,))          # T+1
    
    # Initial values
    if beta_tt_init is None:
        beta_tt[0] = np.linalg.inv(np.eye(k) - F).dot(mu) # kxk * kx1 = kx1
    else:
        beta_tt[0] = beta_tt_init
    if P_tt_init is None:
        P_tt[0] = np.linalg.inv(np.eye(k**2) - np.kron(F,F)).dot(Q.reshape(Q.size, 1)).reshape(k,k) # kxk
    else:
        P_tt[0] = P_tt_init

    # Iterate forwards
    H_view = H[:,:,0]
    for t in range(1,T+1):
        if time_varying_H:
            H_view = H[:,:,t]
        
        # Prediction
        beta_tt1[t] = mu + F.dot(beta_tt[t-1])      # kx1
        P_tt1[t]    = F.dot(P_tt[t-1]).dot(F.T) + Q         # kxk
        y_tt1[t]    = H_view.dot(beta_tt1[t]) + A.dot(z[t]) # nx0
        eta_tt1[t]  = y[t] - y_tt1[t]                       # nx0
        PHT = P_tt1[t].dot(H_view.T)                        # kxk * kxn = kxn
        f_tt1[t]    = H_view.dot(PHT) + R                   # nxk * kxn + nxn = nxn
        f_inv = np.linalg.inv(f_tt1[t])                     # nxn

        # Log-likelihood as byproduct
        ll[t] = -0.5*np.log(2*np.pi*np.linalg.det(f_tt1[t])) - 0.5*eta_tt1[t].T.dot(f_inv).dot(eta_tt1[t])
        
        # Updating
        gain[t] = PHT.dot(f_inv)                            # kxn * nxn = kxn
        beta_tt[t] = beta_tt1[t] + gain[t].dot(eta_tt1[t])
        P_tt[t] = P_tt1[t] - gain[t].dot(H_view).dot(P_tt1[t])
    
    return beta_tt, P_tt, beta_tt1, P_tt1, y_tt1, eta_tt1, f_tt1, gain, ll
