import numpy as np

# 1. Conditional on the parameters and data, generate the states
# Run the Kalman Filter
def sample_states(loadings, obs_cov, phi, state_cov, model, sim):
    # Update the state-space model
    params = np.r_[
        loadings.ravel(), obs_cov, phi, state_cov.ravel()
    ]
    model.update(params)
    #print np.max(np.abs(np.linalg.eigvals(model.transition[:,:,0])))
    # Sample the states from the new posterior
    sim.simulate()
    return sim.simulated_state[:,:-1]

# 2. Conditional on the states, data, and other parameters,
#    generate each set of parameters
def sample_obs_prec(loadings, states, model, gamma):
    """
    Observation Precisions
    
    These are the inverse of the observation variances (the
    diagonal elements of the observation covariance matrix).
    
    They are drawn from a conjugate Gamma posterior on
    equation-by-equation OLS on the observation equation.
    
    The prior for each is Gamma(0.0001, 3), where the first
    parameter is interpreted as a shape parameter and the second
    is interpreted as a rate parameter (the inverse of a scale
    parameter).
    
    The conditional posterior Gamma with the standard
    posterior hyperparameter definition in the independent
    Normal-Gamma case.
    """
    obs_prec = np.zeros((model.k_info,))
    gamma.exog = states[:model.k_posdef,:].T        # Txk
    for i in range(model.k_endog):
        # Update the variance model (i.e. the parameters of a
        # Gamma distribution)
        gamma.endog = model.endog[i]  # Tx1
        
        # For the first k_factors rows, loadings are fixed
        if i < model.k_factors:
            gamma.beta = model.design[i,:model.k_posdef,0]  # kx0
        # For the next k_info - k_factors rows, use estimated loadings
        elif i < model.k_info:
            gamma.beta = loadings[i-model.k_factors]  # kx0
        # For the last k_obs rows, there is, by assumption,
        # nothing to estimate either in loadings or variances
        else:
            break

        # Sample the variance from the new posterior
        obs_prec[i] = next(gamma)
    return obs_prec

def sample_loadings(obs_prec, states, model, normal):
    """
    Factor Loadings
    
    These are the (non-restricted) elements of the design
    matrix.
    
    They are drawn from a conjugate Normal posterior on
    equation-by-equation OLS on the observation equation.
    
    The prior for each is N(0,1)
    
    The conditional posterior for each is Normal with the
    standard posterior hyperparameter definition in the
    independent Normal-Gamma case.
    """
    normal.exog = states[:model.k_posdef,:]  # kxT
    loadings = np.zeros((model.k_info-model.k_factors, model.k_posdef))  # k_info-k_factors x k_posdef
    for i in range(model.k_endog):
        # For the first k_factors rows, there are no
        # loading parameters to estimate
        if i < model.k_factors:
            pass
        # For the next k_info - k_factors rows, sample loadings
        elif i < model.k_info:
            normal.precision = obs_prec[i]  # 1x1
            normal.endog = model.endog[i]  # Tx1
            loadings[i-model.k_factors] = next(normal)
        # For the last k_obs rows, there is, by assumption,
        # nothing to estimate either in loadings or variances
        else:
            break
    return loadings

def sample_state_prec(phi, states, model, wishart):
    """
    State precision matrix
    
    This is the precision matrix of the VAR in the transition
    equation.
    """
#     print states[:model.k_posdef,model.order:].shape
#     print np.hstack([
#         states[:model.k_posdef, model.order-i:-i].T
#         for i in range(1, model.order+1)
#     ]).T.shape
#     print phi.reshape((model.k_posdef, model.k_states)).shape
    # Update the covariance model (Wishart parameters)
    wishart.endog = states[:model.k_posdef,model.order:]
    #wishart.lagged = states[:model.k_posdef,:-model.order]
    wishart.lagged = np.hstack([
        states[:model.k_posdef, model.order-i:-i].T
        for i in range(1, model.order+1)
    ]).T
    wishart.phi = phi.reshape((model.k_posdef, model.k_states))
    
    # Sample the precision from the new posterior
    #print wishart.endog
    #print wishart.lagged
    #print wishart.phi
    out = next(wishart)
    return out

def sample_phi(state_prec, states, model, var, normal):
    """
    Lag Matrix
    
    These are the elements of the transition matrix.
    
    They are drawn from a conjugate Multivariate Normal
    posterior from SUR on the transition equation.
    
    Here the number of equations is equal to the number
    of states: N = K+M. The prior and the posterior are
    calculated using the vectorized notation for the VAR
    coefficient matrix (as in, e.g., Koop and Korobilis,
    2010):
    
    The prior is :math:`\Phi \equiv vec(\phi') \sim MVN(0,\underline{V})`
    (this organization has the first N rows of :math:`\Phi` as the
    coefficients in the first equation - which means that it is the
    first row of the :math:`\phi` matrix.)
    
    where :math:`\underline{V} = diag(\underline{V}_1, ..., \underline{V}_N)` is
    :math:`(N^2 x N^2)` and where :math:`\underline{V}_i ~ (N x N)`
    is based on the Minnesota prior.
    
    In particular, for :math:`\underline{V}_i`, we have:
    - :math:`\underline{V}_{i,ii} = \frac{a_1}{r}` where r denotes lag length
    - :math:`\underline{V}_{i,jj} = \frac{a_2}{r} \frac{\sigma_i^2}{\sigma_j^2}` where r denotes lag length
    
    and where :math:`\sigma_i^2 = s_i^2` from the univariate autoregression
    of order d (here d=1) of the ith state.
    
    The conditional posterior for each is Multivariate
    Normal with the standard posterior hyperparameter
    definition in the independent Normal-Wishart case.
    """
    # # Create the endog and exog matrices
    # endog = states[:,1:]      # k_states x T-1
    # lagged = states[:,:-1].T  # T-1 x k_states
    # exog = np.zeros((model.k_states, model.k_var, model.nobs-model.order))
    # eye = np.eye(model.k_states)
    # for t in range(model.nobs-1):
    #     #exog[:,:,t] = np.kron(eye, lagged[t])
    #     for i in range(model.k_states):
    #         exog[i, i*model.k_states:(i+1)*model.k_states, t] = lagged[t]    
    # # Update the transition matrix model (MVN parameters)
    # normal.endog = endog
    # normal.exog = exog
    # normal.precision = state_prec
    
    # Recalculate the VAR quantites
    var.endog = states[:model.k_posdef]
    var.precision = state_prec
    ZHy, ZHZ = var.quantities
    
    # Update the posterior
    normal._posterior_loc = None
    normal._posterior_scale = None
    normal._posterior_cholesky = None
    normal._ZHZ = ZHZ
    normal._ZHy = ZHy

    #print np.diag(normal.posterior_cholesky)
    
    # Get a draw
    out = next(normal)
    model[model._transition_idx] = np.reshape(
        out, (model.k_posdef, model.k_states)
    )
    eigs = np.max(np.abs(np.linalg.eigvals(model.transition[:,:,0])))
    i = 0
    while(eigs > 0.999):
        i += 1
        out = next(normal)
        model[model._transition_idx] = np.reshape(
            out, (model.k_posdef, model.k_states)
        )
        eigs = np.max(np.abs(np.linalg.eigvals(model.transition[:,:,0])))
        if i % 10 == 0:
            print i
        if i > 1000:
            raise Exception('Stationary phi parameters could not be drawn.')

    return out