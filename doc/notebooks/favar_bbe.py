import os
import numpy as np
import pandas as pd
import tables as tb
from dismalpy import ssm
from dismalpy.ssm import dynamic_factors as df
from dismalpy.stats.rvs import gamma, normal, wishart
from dismalpy.reg import var

class Iteration(tb.IsDescription):
    iteration = tb.Int32Col()
    loadings = tb.Float64Col(shape=(104,6))
    obs_cov = tb.Float64Col(shape=(109))
    phi = tb.Float64Col(shape=(468))
    state_cov = tb.Float64Col(shape=(6,6))
    states = tb.Float64Col(shape=(78,353))

class OriginalIteration(tb.IsDescription):
    iteration = tb.Int32Col()
    loadings = tb.Float64Col(shape=(114,6))
    obs_cov = tb.Float64Col(shape=(119))
    phi = tb.Float64Col(shape=(468))
    state_cov = tb.Float64Col(shape=(6,6))
    states = tb.Float64Col(shape=(78,511))


def get_storage(file='favar_bbe.h5', replace=False, original=False):
    if replace or not os.path.isfile(file):
        filters = tb.Filters(complevel=9, complib='blosc', fletcher32=True)
        filename = "favar_bbe.h5" if not original else "favar_bbe_original.h5"
        h5 = tb.open_file(filename, mode="w", title="FAVAR BBE Gibbs Iterations", filters=filters)
        group = h5.create_group(h5.root, 'iterations', 'Gibbs Sampling iterations')
        klass = Iteration if not original else OriginalIteration
        table = h5.create_table(h5.root.iterations, 'data', klass, 'Iteration data')
        table.cols.iteration.create_csindex()
    else:
        filename = "favar_bbe.h5" if not original else "favar_bbe_original.h5"
        h5 = tb.open_file(filename, mode="a", title="FAVAR BBE Gibbs Iterations")
        table = h5.root.iterations.data
    return h5, table

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

# Define the Gibbs-Sampling iterations
def sample(model, G1, G, replace=False):
    # Create the samplers
    sampler_state = model.simulation_smoother(simulation_output=ssm.SIMULATION_STATE)
    sampler_state._simulation_smoother.simulated_model.subset_design = True
    sampler_state._simulation_smoother.simulated_model.companion_transition = True
    
    sampler_obs_prec = gamma.Gamma(prior_obs_prec_shape / 2., 2. / prior_obs_prec_rate, preload=G1+G)
    sampler_loadings = normal.Normal(loc=prior_loading_mean, scale=prior_loading_var, preload=G1+G)
    sampler_state_prec = wishart.Wishart(prior_state_prec_df, prior_state_prec_scale, preload=G1+G)
    sampler_phi = normal.Normal(loc=prior_state_mean, scale=prior_state_var, preload=G1+G)
    sampler_phi._use_posterior = True
    
    # Create a VAR
    var_phi = var.VAR(nobs=model.nobs, k_endog=model.k_posdef, order=model.order)
    
    # Storage
    h5, table = get_storage(replace=replace)
    
    try:
        # Initial values
        if len(table.col('iteration')) == 0:
            latest = 0
            start_params = model.start_params
            loadings = np.reshape(
                start_params[model._params_loadings],
                (model.k_info - model.k_factors, model.k_posdef)
            )
            obs_cov = start_params[model._params_obs_cov]
            obs_prec = 1. / obs_cov[0]
            phi = start_params[model._params_transition]
            state_cov = np.reshape(
                start_params[model._params_state_cov],
                (model.k_posdef, model.k_posdef)
            )
            state_prec = np.linalg.inv(state_cov)
        else:
            latest = table.colindexes['iteration'][-1]
            iteration, loadings, obs_cov, phi, state_cov, states = table[latest]
            obs_prec = 1. / obs_cov
            state_prec = np.linalg.inv(state_cov)

        # Run the iterations
        iteration = table.row
        for i in range(G1+G):
            # Unobserved states
            states = sample_states(loadings, obs_cov, phi, state_cov, model, sampler_state)

            # Observation equation (OLS)
            obs_prec = sample_obs_prec(loadings, states, model, sampler_obs_prec)
            obs_cov = 1. / obs_prec
            loadings = sample_loadings(obs_prec, states, model, sampler_loadings)

            # Transition equation (VAR)
            state_prec = sample_state_prec(phi, states, model, sampler_state_prec)
            state_cov = np.linalg.inv(state_prec)
            phi = sample_phi(state_prec, states, model, var_phi, sampler_phi)

            # Store the data
            iteration['iteration'] = latest + i
            iteration['states'] = states
            iteration['obs_cov'] = obs_cov
            iteration['loadings'] = loadings
            iteration['state_cov'] = state_cov
            iteration['phi'] = phi
            iteration.append()

            if i % 100 == 0:
                table.flush()
                print i
    finally:
        h5.close()


if __name__ == '__main__':
    pass

    # columns = pd.read_csv('data/favar_bgm_columns.csv')
    # columns.index = columns.column
    # del columns['column']
    # columns.sort(inplace=True)
    # dates = pd.date_range('1976-02','2005-06', freq='MS')

    # # Get the data
    # raw = pd.read_csv('data/data76.csv', header=None)
    # raw.columns = columns.id.tolist()
    # raw.index = dates

    # # Only take the BBE dataset
    # raw = raw.iloc[:,:110]

    # # Standardize the variables
    # dta = (raw - raw.mean()) / raw.std()

    # # Separate into observed economic variables and background variables
    # observed = dta['FYFF']
    # informational = dta.drop('FYFF', axis=1)

    # # Construct the model
    # mod = df.FAVAR(observed, informational, k_factors=5, order=13)

    # # Set to use the univariate filter with observation collapsing
    # mod.filter_method = ssm.FILTER_COLLAPSED | ssm.FILTER_UNIVARIATE
    # mod._initialize_representation()
    # mod._statespace.subset_design = True
    # mod._statespace.companion_transition = True
    # mod.endog.shape

    # # Observation variances prior: Gamma hyperparameters
    # prior_obs_prec_shape = 1e-4 # 2*alpha
    # prior_obs_prec_rate = 3    # 2*beta
    # prior_obs_prec_exp = ((prior_obs_prec_shape/2.) / (prior_obs_prec_rate/2.))
    # print 'Prior expected value of obs. variances is %.5f' % (1./prior_obs_prec_exp)

    # # Observation loadings prior: Normal hyperparameters
    # prior_loading_mean = np.zeros(mod.k_posdef)
    # prior_loading_var = np.eye(mod.k_posdef)

    # # State precision prior
    # prior_state_prec_df = mod.k_posdef
    # prior_state_prec_scale = np.eye(mod.k_posdef)

    # # State prior
    # prior_state_mean = np.zeros(mod.k_var)
    # prior_state_var = np.eye(mod.k_var)*0.01

    # # Iterations
    # G1 = 2000
    # G = 20000
    # tic = time.time()
    # sample(mod, G1, G, True)
    # print time.time() - tic
