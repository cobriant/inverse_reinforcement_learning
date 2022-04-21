#' ---
#' title: "IRL from Sample Trajectories"
#' author: "Colleen O'Briant"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---
#+ message = FALSE
library(tidyverse)

#' Here I continue using the example from `example2.R`. In this case however,
#' I deal with the more realistic case where the optimal policy is unknown, but
#' trajectories under the optimal policy (`data`) are available.
#'
#' The strategy for estimation is as follows:
#' 
#' 1. Given the `data`, we can estimate the value function $\hat{V}^{\pi}(S_0)$ 
#'    as a linear combination of value functions of each gaussian basis function 
#'    $\hat{V}_i^{\pi}(S_0)$ on average if the reward was only $R = \phi_i$. 
#'    That is, 
#'    
#'    $$\hat{V}_i^{\pi}(S_0) = E[\phi_i(S_0) + \beta \phi_i (S_1) + \beta^2 \phi_i (S_2)] + ...$$
#'    
#'    and
#'    
#'    $$\hat{V}^{\pi}(S_0) = \alpha_1 \hat{V}_1^{\pi}(S_0) + \hat{V}_2^{\pi}(S_0) + ...$$
#'
#' 2. Generate a random policy `pi_rand` and generate some trajectories 
#'    `traj_rand` of the data generating process using `pi_rand`. Use 
#'    `traj_rand` to estimate another value function, following the same process 
#'    as in step 1.
#'    
#' 3. Now we have two value functions (one for `data` and one for `traj_rand`).
#'    We can use those two value functions to make an estimate for the 
#'    $\alpha_i$'s for the true reward function using linear programming:
#'    
#'    $\text{maximize} \sum (\hat{V}^{\pi^*}(S_0) - \hat{V}^{\pi_i}(S_0))$
#'
#'    $\text{s.t.} |\alpha_i| \le 1, i = 1, ..., d$
#'    
#' 4. Linear programming will give us a solution for the $\alpha_i$'s, which we
#'    can use to calculate the reward function R(s) and the policy function it 
#'    imples, which we'll call $\pi_{k+1}$. But that solution is only a rough 
#'    approximation because it only factors in the difference between 
#'    $\hat{V}^{\pi_star}(S_0)$ and a value function from one randomly chosen 
#'    policy. So we can iterate and watch R(s) for convergence: use $\pi_{k+1}$
#'    to form more trajectories and another value function. In the LP step for 
#'    step 3, maximize the sum of the differences of $\hat{V}^{\pi_star}(S_0)$ 
#'    with the value functions from all the policies seen thus far.
#'    
#'
#' # 1 Create `data` using a simulation under the optimal policy $\pi^*$
#' 
#' Let $\pi^*$ be "stay" if s < 0 and "switch" if s >= 0.

s <- runif(n = 1, min = -1, max = 1) # initial state randomly chosen

pi_star <- function(s) if_else(s < 0, 1, -1) # 1 for stay; -1 for switch

noise <- function() runif(n = 1, min = -.5, max = .5)

truncate <- function(s) {
  case_when(s > 1 ~ 1,
            s < -1 ~ -1,
            TRUE ~ s)
}

step <- function(s, policy) {
  (policy(first(s))*first(s) + noise()) %>% 
    truncate()
}

# Trajectory of 20 steps following pi_star with a random initial state:
reduce(1:19, 
       function(x, y) c(step(x, pi_star), x), 
       .init = runif(n = 1, min = -1, max = 1))

# 20 trajectories of 20 steps each
monte_carlo_sim <- function(policy) {
  map(1:20, 
      function(...) {
        reduce(1:19, 
               function(x, y) c(step(x, policy), x), 
               .init = runif(n = 1, min = -1, max = 1))
      })
}

data <- monte_carlo_sim(pi_star)

#' $\hat{V_i}^{\pi^*}$ : the average empirical return on each of the m
#' trajectories if the reward had been $R = \phi_i$.
#'
#' $\hat{V_1}^{\pi^*}$: take $R_1$ = dnorm(s, mean = -1, sd = 2/22), discount 
#' future rewards by a factor of 0.9, and sum rewards over time. $\hat{V_1}^{\pi^*}$
#' will be the average of these (`data[[1]]` to `data[[100]]`):

sum(dnorm(data[[1]], mean = -1, sd = 2/22)*(.9^(0:99)))
sum(dnorm(data[[2]], mean = -1, sd = 2/22)*(.9^(0:99)))
sum(dnorm(data[[3]], mean = -1, sd = 2/22)*(.9^(0:99)))

# Which is this:

map_dbl(
  1:20, 
  ~ sum(dnorm(data[[.x]], mean = -1, sd = 2/22)*(.9^(0:99)))
  ) %>%
  mean()

#' $\hat{V}^{\pi} = \alpha_1 \hat{V_1}^{\pi}(s_0) + ... + \alpha_d \hat{V_d}^{\pi}(s_0)$

calc_value_function_estimate <- function(data) {
  map_dbl(
    seq(-1, 1, length = 21), 
    function(y) {
      map_dbl(1:20, 
              ~ sum(dnorm(data[[.x]], mean = y, sd = 2/22)*(.9^(0:99)))) %>%
        mean()
      })
}

value_pi_star <- calc_value_function_estimate(data)

#' # 2 Generate a random policy `pi_rand`, 
#' 
#' trajectories `traj_rand`, and value function `value_pi_rand`.
#' For simplicity I'll only consider polices where you "stay" until some s 
#' inflection, and then "switch" from then on out, /or/ "switch" until some s 
#' inflection, and then "stay" from then on out.
#' 

set.seed(1234)
shape <- sample(c(-1, 1), size = 1)
inflection <- sample(seq(-1, 1, length = 21), size = 1)
pi_rand <- function(s) if_else(s < inflection, -shape, shape)

# 20 trajectories of 20 steps each
traj_rand <- monte_carlo_sim(pi_rand)
value_pi_rand <- calc_value_function_estimate(traj_rand)

#' # 3 Use an LP solver to use v_pi and value_pi_rand to estimate $\alpha_i$'s
#' 

# Standard LP form: minimize c'x
#                   s/t Ax <= b
#                       x >= 0
# x: (alpha_1, alpha_2, ..., alpha_21)
# c: v_pi - value_pi_rand
# constraints: alpha_i <= 1
# A = diag(21)
# b: rep(1, 21)

alpha <- lpSolve::lp(
  direction = "max",
  objective.in = matrix(value_pi_star - value_pi_rand, nrow = 21),
  const.mat = diag(21),
  const.dir = rep("<=", 21),
  const.rhs = matrix(rep(1, 21), nrow = 21)
)$solution

#' # 4 Use alphas to form R(s); use R(s) to estimate $\pi^*$
#' 
#' Then loop back to step 2, forming a new policy but using the reward function
#' from the previous iteration this time. Using the new policy, calculate a set
#' of trajectories and an estimate for the value function. In the LP step, use 
#' all the value functions we've calculated to form new approximations of the 
#' alphas until the alphas (and the reward functions) converge.
#' 

# R(s) = sum(alpha_i*phi_i), where phi_i are the 21 gaussian basis functions

print(alpha)

bind_rows(
  tibble(r = alpha,
         x = seq(-1, 1, length = 21),
         category = "estimate"
  ),
  tibble(r = c(rep(1, 10), rep(0, 11)),
         x = seq(-1, 1, length = 21),
         category = "true value"
  )
  ) %>%
  ggplot(aes(x = x, y = r, color = category)) +
  geom_jitter(height = 0.02) +
  ggtitle("The approximated reward function is already 
          close to the true one")

reward_function <- function(s, alpha) {
  sum(
    alpha * map_dbl(seq(-1, 1, length = 21), 
                    ~ dnorm(s, mean = .x, sd = 2/22))
  )
}

#' $\pi_{k+1}$ maximizes R(s). In this situation it's straightforward that
#' you should stay if you're receiving rewards and switch if not.

lp_irl_sample_trajectories <- function(data, printing = T) {
  
  value_functions <- calc_value_function_estimate(data) %>% list()
  alpha <- rep(0, 21) #init
  
  # Generate a random policy
  shape <- sample(c(-1, 1), size = 1)
  inflection <- sample(seq(-1, 1, length = 21), size = 1)
  pi1 <- function(s) if_else(s < inflection, -shape, shape)
  
  for (i in 1:20) {
    
    # Generate trajectories with pi_{k + 1}
    traj1 <- monte_carlo_sim(pi1)
    
    # Calculate the value function of that random policy and add it to
    # a list of value functions, the last of which is V^pi*
    value_functions <- calc_value_function_estimate(traj1) %>%
      list() %>%
      append(value_functions)
    
    # Build the c matrix by iterating over value_functions:
    objective_in <- rep(0, 21) # init
    
    for (j in 1:(length(value_functions) - 1)) {
      objective_in <- (last(value_functions) - value_functions[[j]]) +
        objective_in
    }
    
    # Use the LP solver to estimate alphas 
    alpha_next <- lpSolve::lp(
      direction = "max",
      objective.in = matrix(objective_in, nrow = 21),
      const.mat = diag(21),
      const.dir = rep("<=", 21),
      const.rhs = matrix(rep(1, 21), nrow = 21)
    )$solution
    
    # Check for convergence in alpha after minimum iterations = 5
    if (i >= 5 & max(abs(alpha_next - alpha)) < .1) {
      return(list(alpha_next, value_functions))
    }
    
    if (printing == T) {
      print(paste0("Iteration ", i))
      print(alpha_next)
    }

    alpha <- alpha_next
    
    # New policy comes from the previous iteration's reward function (alphas)
    pi1 <- function(s) if_else(reward_function(s, alpha) > .5, 1, -1)
  }
}

#' # 5 Run the algorithm 5 times and compare to the optimal policy

alpha1 <- lp_irl_sample_trajectories(data, print = F)[[1]]
alpha2 <- lp_irl_sample_trajectories(data, print = F)[[1]]
alpha3 <- lp_irl_sample_trajectories(data, print = F)[[1]]
alpha4 <- lp_irl_sample_trajectories(data, print = F)[[1]]
alpha5 <- lp_irl_sample_trajectories(data, print = F)[[1]]

bind_rows(
  tibble(r = c(alpha1, alpha2, alpha3, alpha4, alpha5),
         x = rep(seq(-1, 1, length = 21), 5),
         category = "estimate"
         ),
  tibble(r = c(rep(1, 10), rep(0, 11)),
         x = seq(-1, 1, length = 21),
         category = "true value"
  )
) %>%
  ggplot(aes(x = x, y = r, color = category)) +
  geom_jitter(height = 0.02)
