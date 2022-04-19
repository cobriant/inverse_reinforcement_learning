#' ---
#' title: "IRL in Large State Spaces"
#' author: "Colleen O'Briant"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---
#+ message = FALSE
library(tidyverse)
library(testthat)
source("lp_irl_function.R")

#' Take the example from `example1.R`, but consider a case where the state space
#' is the infinite set of real numbers between -1 and 1 inclusive. The strategy
#' for estimating the reward function R(s) is largely the same as for the 
#' finite state space case with 2 differences:
#' 
#' 1) I'll take a subsample `s0` of states in S to create the probability 
#'    transition matrices:
#'    

s0 <- seq(-1, 1, length = 21)

#'    
#' 2) I'll estimate the reward function as a linear combination of Gaussian
#'    basis functions $\phi_d$, each centered on an element of `s0` with 
#'    `sd = 2/22`. $R(s) = \alpha_1 \phi_1(s) + \alpha_2 \phi_2(s) + ... + \alpha_{21} \phi_{21}(s)$
#'    IRL will estimate the $\alpha_i$'s.
#' 

map_dfr(
  seq(-1, 1, length = 21),
  function(cent) { tibble(center = cent, 
                        x = seq(-1.5, 1.5, length = 500),
                        y = dnorm(x, mean = center, sd = 2/22))
  }
  ) %>%
  mutate(center = as.factor(center)) %>%
  ggplot(aes(x = x, y = y, color = center)) +
  geom_line() +
  ggtitle("21 Evenly Spaced Gaussian Basis Functions on [-1, 1]")

#' 
#' Let the true reward function be 1 if $s < 0$ and 0 if $s \ge 0$.
#' 
#' Available actions will once again be:
#' 
#' * 1: "stay"
#' * 2: "randomize"
#' * 3: "switch"
#' 
#' But this time, noise distributed uniform [-.5, .5] is introduced. So if 
#' you're in state -0.5 and you "stay", your state next period is distributed
#' uniform [-1, 0]. That is, s' = s + noise. If you "switch", -0.5 becomes 0.5 
#' and then noise is added: your state next period is distributed uniform 
#' [0, 1]. So "switching" results in s' = -s + noise. If you "randomize",
#' w.p. 0.5, your state next period is uniform [-1, 0] and w.p. 0.5, your state
#' next period is uniform [0, 1]. So "randomizing" yields s' = s + noise half of 
#' the time and -s + noise half the time.
#' 

p_ssa <- array( # init s0 has length 21; there are 3 actions available
  data = rep(0, 21*21*3),
  dim = c(21, 21, 3)
)

# action 1: "stay"
for (i in 1:21) {
  left_index <- if_else(i - 5 > 0, i - 5, 1)
  right_index <- if_else(i + 5 > 21, 21, i + 5)
  possible_s2 <- s0[left_index:right_index] #all possible values for s'
  p_ssa[i, , 1] <- s0 %in% possible_s2 / length(possible_s2)
}

# action 3: "switch"
for (i in 1:21) {
  j <- 22 - i # s --> -s
  left_index <- if_else(j - 5 > 0, j - 5, 1)
  right_index <- if_else(j + 5 > 21, 21, j + 5)
  possible_s2 <- s0[left_index:right_index] #all possible values for s'
  p_ssa[i, , 3] <- s0 %in% possible_s2 / length(possible_s2)
}

# action 2: "randomize"
for (i in 1:21) {
  # First take case where coin toss is to stay. Multiply denominator by 2 
  # because this only happens half the time.
  left_index <- if_else(i - 5 > 0, i - 5, 1)
  right_index <- if_else(i + 5 > 21, 21, i + 5)
  possible_s2 <- s0[left_index:right_index] #all possible values for s'
  p_ssa[i, , 2] <- s0 %in% possible_s2 / (length(possible_s2)*2)
  # Then take the case where the coin toss is to switch. Multiply denominator
  # by 2 and *add* to current p_ssa because there are 2 ways to get to any 
  # state: by staying and landing on it or by switching and landing on it.
  j <- 22 - i # s --> -s
  left_index <- if_else(j - 5 > 0, j - 5, 1)
  right_index <- if_else(j + 5 > 21, 21, j + 5)
  possible_s2 <- s0[left_index:right_index] #all possible values for s'
  p_ssa[i, , 2] <- p_ssa[i, , 2] + 
    (s0 %in% possible_s2 / (length(possible_s2)*2))
}

#' Optimal policy:
#' 
#' If s < 0, stay. If s >= 0, switch.

sol <- lp_irl(
  p_ssa = p_ssa, 
  optimal_policy = c(rep(1, 10), rep(3, 21 - 10)),
  beta = .9,
  R_max = 1
)

sol$reward_function

#' Reward function: IRL estimated it right on. We want to think of these as
#' the alphas mentioned above, so we can extrapolate the reward of a state 
#' between elements of s0:

R <- function(s) {
  dnorm(s, mean = -1, sd = 2/22) +
    dnorm(s, mean = -.9, sd = 2/22) +
    dnorm(s, mean = -.8, sd = 2/22) +
    dnorm(s, mean = -.7, sd = 2/22) +
    dnorm(s, mean = -.6, sd = 2/22) +
    dnorm(s, mean = -.5, sd = 2/22) +
    dnorm(s, mean = -.4, sd = 2/22) +
    dnorm(s, mean = -.3, sd = 2/22) +
    dnorm(s, mean = -.2, sd = 2/22) +
    dnorm(s, mean = -.1, sd = 2/22)
}

tibble(x = seq(-1, 1, length = 100), y = R(x)) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() +
  ggtitle("Reward Function")

#' 
#' The cost of a single-step deviation from the optimal policy is highest at 
#' the ends of the s distribution: if s = -1, you want to stay, and switching
#' means you earn 0 with certainty in the next period. Likewise if s = 1, you
#' want to switch, and staying means you earn 0 with certainty in the next 
#' period. Toward the middle, the difference between behaving optimally and not
#' matters less because noise is the dominant factor that determines whether
#' or not you'll earn a reward.
#' 

tibble(x = seq(-1, 1, length = 21), y = sol$cost_singlestep_deviation) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() +
  ggtitle("Cost of a single-step deivation from optimal_p")

#' Value function: the best state to be in is -1. Then the value falls as you 
#' get closer to zero. The value rises again as you approach 1 because from 1, 
#' it's easy to go to -1 with a switch.

tibble(x = seq(-1, 1, length = 21), y = sol$value_function) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() +
  ggtitle("Value Function")
