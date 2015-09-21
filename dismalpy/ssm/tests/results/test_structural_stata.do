// Dataset
insheet using ../../../../datasets/macrodata/macrodata.csv, clear
gen lnunemp = log(unemp)

// Get the time variable
gen tq = yq(year, quarter)
format tq %tq
tsset tq

// Irregular
ucm unemp, model(ntrend)

// Irregular + Deterministic level
ucm lnunemp, model(dconstant)

// Local level

// Random walk
ucm lnunemp

// Random walk: State space model
constraint 1 [lnunemp]u1 = 1
constraint 2 [u1]L.u1 = 1

sspace (u1 L.u1, state noconstant) ///
       (lnunemp u1, noconstant noerror), ///
       constraints(1/2) covstate(diagonal)
