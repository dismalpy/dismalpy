library(foreign)
library(FKF)

# Observations
data <- read.dta('results/wpi1.dta')
dwpi = diff(data$wpi)

# True parameters
params <- c(
  .5270715, .0952613, .2580355, # AR
  .5307459 # sigma
)

# Dimensions
n = 1
k = 3

# Measurement equation
H = matrix(rep(0, n*k), nrow=n) # design
H[1,1] = 1
obs_intercept = matrix(rep(0,n), nrow=n)
R = matrix(rep(0, n^2), nrow=n) # obs_cov

# Transition equation
mu = matrix(rep(0, k), nrow=k)
F = matrix(rep(0, k^2), nrow=k) # transition
F[2,1] = 1
F[3,2] = 1

# Q = G Q_star G'
Q = matrix(rep(0, k^2), nrow=k) # selected_state_cov

# Update matrices with given parameters
F[1,1] = params[1]
F[1,2] = params[2]
F[1,3] = params[3]

Q[1,1] = params[4]^2

# Initialization: Stationary priors
initial_state = c(solve(diag(k) - F) %*% mu)
initial_state_cov = solve(diag(k^2) - F %x% F)  %*% matrix(c(Q))
dim(initial_state_cov) <- c(k,k)

# Filter
ans <- fkf(a0=initial_state, P0=initial_state_cov,
           dt=mu, ct=obs_intercept, Tt=F, Zt=H,
           HHt=Q, GGt=R, yt=rbind(dwpi))
# ans$at  # predicted states
# ans$att # filtered states

# Get smoothed states
arima <- makeARIMA(params[1:3],c(),c())
arima$T <- t(arima$T)
arima$V[1,1] <- params[4]^2
arima$Pn <- initial_state_cov
filtered <- KalmanRun(dwpi, arima) # filtered states
smoothed <- KalmanSmooth(dwpi, arima) # smoothed states

# Output data
predicted = t(ans$at)
output <- data.frame(
  sp1=predicted[1:123,1],
  sp2=predicted[1:123,2],
  sp3=predicted[1:123,3],
  sf1=filtered$states[,1],
  sf2=filtered$states[,2],
  sf3=filtered$states[,3],
  sm1=smoothed$smooth[,1],
  sm2=smoothed$smooth[,2],
  sm2=smoothed$smooth[,3]
)
write.csv(output, file='results_wpi1_ar3_R.csv', row.names=FALSE)