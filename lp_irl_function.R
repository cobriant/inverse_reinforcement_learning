#' ---
#' title: "Linear Programming IRL Function"
#' author: "Colleen O'Briant"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---
#+ message = FALSE
library(tidyverse)
library(testthat)

lp_irl <- function(p_ssa, optimal_policy, beta, R_max) {
  
  # Args:
  # p_ssa: transition probability array that is (n x n x k)
  expect_equal(dim(p_ssa)[1], dim(p_ssa)[2])
  # optimal_policy: optimal policies at each state (n x 1)
  expect_length(optimal_policy, dim(p_ssa)[1])
  # beta: discount factor
  expect_lt(beta, 1)
  # R_max: upper bound for reward function (scalar)
  expect_length(R_max, 1)
  
  n <- dim(p_ssa)[1]
  k <- dim(p_ssa)[3]
  
  # Under the optimal policy, the transition probability matrix is p_{a1}:
  p_a1 <- matrix(rep(0, n*n), nrow = n)
  for (i in 1:n) {
    p_a1[i, ] <- p_ssa[i, , optimal_policy[i]]
  }
  expect_setequal(rowSums(p_a1), 1)
  
  # Helper function:
  drop_in_v <- function(a, i){
    # Drop in value from initially choosing action a instead of a_1 in state i,
    # but then behaving according to a_1 otherwise = drop_in_v(a, i) %*% R
    (p_a1[i, ] - p_ssa[i, , a]) %*% solve(diag(n) - beta*p_a1)
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
  C_mat <- matrix(c(rep(1, n), rep(0, n)), nrow = (n + n))
  
  # A includes 3 types of constraints:
  # 1. The constraint on z_i already mentioned: 
  #    z_i <= drop_in_r(a, i) %*% R, \forall a \in A /a_star
  #    This is n*(k-1) constraints.
  # 2. a_star must be optimal, so
  #    drop_in_r(a, i) %*% R >= 0 \forall a \in A.
  #    That is, choosing any action other than a_star yeilds a nonnegative
  #    drop in rewards.
  #    This is n*k constraints.
  # 3. R_i <= R_max for all i.
  #    This is n constraints.
  # In total, A will have n*(k-1) + n + n*k rows and 2n columns.
  
  A_mat <- c() # init
  
  # Building A in 3 parts:
  
  # 1. z_i <= drop_in_r(a, i) %*% R, \forall a \in A /a_star
  #    z_i - drop_in_r(a, i) %*% R <= 0
  #    A[i, ] = 1 - drop_in_r(a, i) %*% R
  for(i in 1:n) {
    for(a in (1:k)[-optimal_policy[i]]) {
      tmp <- matrix(c(rep(0, n), -1*drop_in_v(a, i)), nrow = 1, byrow = T)
      tmp[1, i] <- 1
      A_mat <- A_mat %>% rbind(tmp)
    }
  }
  
  # 2. p_a1 must be optimal, so drop_in_r(a, i) %*% R >= 0 \forall a \in A \a*.
  for(i in 1:n) {
    for(a in (1:k)[-optimal_policy[i]]) {
      tmp <- matrix(c(rep(0, n), drop_in_v(a, i)), nrow = 1, byrow = T)
      A_mat <- A_mat %>% rbind(tmp)
    }
  }
  
  # 3. R_i <= R_max for all i
  # R_max will go in b; we just need 1's for each R_i here.
  for(i in 1:n) {
    tmp <- matrix(c(rep(0, n + n)), nrow = 1, byrow = T)
    tmp[1, n + i] <- 1
    A_mat <- A_mat %>% rbind(tmp)
  }
  
  B_mat <- matrix(c(rep(0, 2*n*(k-1)), rep(R_max, n)), ncol = 1)
  
  # Now to solve:
  
  sol <- lpSolve::lp(
    direction = "max",
    objective.in = C_mat,
    const.mat = A_mat,
    const.dir = c(rep("<=", n*(k-1)), rep(">=", n*(k-1)), rep("<=", n)),
    const.rhs = B_mat 
  )
  
  value_function <- solve(
    diag(n) - beta * p_a1
    ) %*% matrix(sol$solution[(n + 1):(2*n)], nrow = n)
  
  return(list(reward_function = sol$solution[(n + 1):(2*n)],
              value_function = value_function,
              cost_singlestep_deviation = sol$solution[1:n]
  ))
}


# Testing the function with lp_irl_example.R numbers:
# should output 5, 5, 10, 0.

lp_irl(
  p_ssa = array(
    data = c(1, 0, 0, 1, .5, .5, .5, .5, 0, 1, 1, 0),
    dim = c(2, 2, 3)
  ), # action 1: stay; action 2: randomize; action 3: switch
  optimal_policy = c(1, 3),
  beta = .9,
  R_max = 10
)
