#' ---
#' title: "Linear Programming IRL Example"
#' author: "Colleen O'Briant"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---
#+ message = FALSE
library(tidyverse)

#' Take a simple example where there are 2 states (S = 1 or S = 2), state 1 is
#' preferable to state 2, and there are 3 actions available to the agent:
#' "stay", "randomize", or "switch". If the agent chooses to stay, their state
#' in the next period will be the same as their state this period. If the agent
#' chooses to switch, their state will change. If the agent chooses to 
#' randomize, their state will stay w.p. 0.5 and change w.p. 0.5. So the state
#' transition array is this:
#' 

p_ssa <- array(
  data = c(1, 0, 0, 1, .5, .5, .5, .5, 0, 1, 1, 0),
  dim = c(2, 2, 3)
) # action 1: stay; action 2: randomize; action 3: switch

print(p_ssa)

#' State 1 is preferable to state 2 because state 1 earns the agent higher 
#' rewards. IRL will calculate the reward function, but we need to put an upper
#' bound on it (lower bound will be 0, which will be implicit in LP solver).
Rmax <- 10

#' Since state 1 is preferable to state 2, the optimal policy is to stay if 
#' you're in state 1 and switch if you're in state 2. This keeps you in state
#' 1 as much as possible.
optimal_p <- c(1, 3)
#' Under the optimal policy, the transition probability matrix is $p_{a_1}$:
p_a1 <- rbind(p_ssa[1, , optimal_p[1]], p_ssa[2, , optimal_p[2]])
print(p_a1)

n <- dim(p_ssa)[1] # Number of states i = 1, ..., N
k <- dim(p_ssa)[3] # Number of actions a = 1, ..., K

# Drop in value from initially choosing action a instead of a_1 in state i,
# but then behaving according to a_1 otherwise = drop_in_v(a, i) %*% R
drop_in_v <- function(a, i){
  (p_a1[i, ] - p_ssa[i, , a]) %*% solve(diag(n) - .9*p_a1)
}

# Standard LP form: minimize c'x
#                   s/t Ax <= b
#                       x >= 0

# x is (z_1, z_2, ..., z_n, R_1, R_2, ..., R_n)
# where z_i is the drop in value from initially choosing the second-best
# option in state i = drop_in_v(a_2, i) %*% R.

# c is 1 x (N + N) with 1's for the first N and 0s for the last N:
# maximize the sum of the z_i's to rule out degenerate solutions like
# R_i = 0 for all i.
C_mat <- matrix(c(rep(1, n), rep(0, n)), nrow = 4)

B_mat <- matrix(c(rep(0, 4), rep(0, 6), 10, 10), ncol = 1)

A_mat <- matrix(
  c(
    # z_i is the drop in value from the second-best option,
    # so z_i <= drop_in_value(any a not a_1, i) * R
    1, 0, - drop_in_v(2, 1), # z_1 <= drop_in_v(action 2 in state 1) * R
    1, 0, - drop_in_v(3, 1), # z_1 <= drop_in_v(action 3 in state 1) * R
    0, 1, - drop_in_v(1, 2), # z_2 <= drop_in_v(action 1 in state 2) * R
    0, 1, - drop_in_v(2, 2), # z_2 <= drop_in_v(action 2 in state 2) * R
    
    # a_1 is optimal, so drop_in_v(any action, state i) %*% R >= 0
    0, 0, drop_in_v(1, 1), 
    0, 0, drop_in_v(2, 1), 
    0, 0, drop_in_v(3, 1), 
    0, 0, drop_in_v(1, 2), 
    0, 0, drop_in_v(2, 2), 
    0, 0, drop_in_v(3, 2), 
    
    # R <= R_max
    0, 0, 1, 0, # <= R_max
    0, 0, 0, 1 # <= R_max
  ), nrow = 12, byrow = T
)

sol <- lpSolve::lp(
  direction = "max",
  objective.in = C_mat,
  const.mat = A_mat,
  const.dir = c(rep("<=", 4), rep(">=", 6), rep("<=", 2)),
  const.rhs = B_mat
)

print(sol$solution)

#' Interpretation:
#' 
#' $z_i$ = drop in $V^\pi$ from picking the second-best option in period 1 given
#' period 1 takes on state i, and then acting according to the optimal policy
#' from then on out.

print(sol$solution[1:2])

#' That is, suppose the initial state is 1, and you choose to randomize instead
#' of staying. Then your expected payoff in the next period is 5 instead of 10.
#' If you behave otherwise optimally, your total change in rewards by making
#' that single step deviation from the optimal policy is 10 - 5 = 5. The same
#' is true if the initial state was 2.
#'
#' The reward function is:

print(sol$solution[3:4])

#' That is, R(1) = 10 ($= R_{max}$) and R(2) = 0 ($= R_{min}$). It is verified 
#' that state 1 is preferable to state 2.
#' 
#' Also, we can find $V^\pi(s_1) = E[R(s_1) + \beta R(s_2) + \beta^2 R(s_3) + ... | \pi]$
#' $= (I - \beta P_{a_1})^{-1} R$
#' 

solve(diag(2) - .9 * p_a1) %*% matrix(c(10, 0), nrow = 2)

#' Which we can verify:
#' Take the initial state as 1. Then the agent receives 10 in state 1 and can
#' continue to get 10 each period after that by acting in accordance with the
#' optimal policy. So $V^\pi(1) = E[10 + \beta 10 + \beta^2 10 + ... ]$.
#' Solve this infinite sum by noting that for $\beta = .9$, 
#' 
#' $.9 V(1) = V(1) - 10$,
#' 
#' so $10 = .1 V(1)$ and $V(1) = 100$.
#' 
#' Take the initial state as 2. Then the agent receives 0 in state 1 but can
#' continue to get 10 each period after that by acting in accordance with the
#' optimal policy. So $V^\pi(2) = E[\beta 10 + \beta^2 10 + ... ]$.
#' Solve this infinite sum by noting that for $\beta = .9$, 
#' 
#' $.9 V(2) = V(2) - 9$,
#' 
#' so $9 = .1 V(2)$ and $V(2) = 90$.
#' 