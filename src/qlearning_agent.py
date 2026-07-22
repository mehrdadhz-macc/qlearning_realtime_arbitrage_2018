"""Tabular Q-learning agent (paper Section III-D, Algorithm 1).

Q(s,a) <- (1-alpha) Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') ]   (Eq. 7)

Algorithm 1's initialization line is printed as "alpha = 0.5, alpha = 0.9,
epsilon = 0.9" -- the repeated alpha is a typo in the paper; alpha (learning
rate) and gamma (discount factor) are two distinct symbols used everywhere
else in the paper, so we read this as alpha=0.5, gamma=0.9, epsilon=0.9 and
default to that.

Epsilon is used as a fixed (non-decaying) exploration probability, exactly as
Algorithm 1 specifies -- there's no decay schedule in the paper. Fixed
epsilon=0.9 means the greedy action is taken only 10% of the time throughout
training; this looks unusual for a policy meant to be deployed, but it's what
Algorithm 1 literally states, and evaluate.py always runs the FINAL Q-table
purely greedily (eps=0) regardless of the epsilon used during training, so
training-time exploration doesn't leak into the reported evaluation policy.
"""

import numpy as np


class QLearningAgent:
    def __init__(self, n_price_bins, n_energy_bins, n_actions=3,
                 alpha=0.5, gamma=0.9, epsilon=0.9, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q = np.zeros((n_price_bins, n_energy_bins, n_actions))
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        price_bin, energy_bin = state
        return int(np.argmax(self.q[price_bin, energy_bin]))

    def update(self, state, action, reward, next_state):
        price_bin, energy_bin = state
        current = self.q[price_bin, energy_bin, action]
        if next_state is None:
            target = reward
        else:
            next_price_bin, next_energy_bin = next_state
            target = reward + self.gamma * np.max(self.q[next_price_bin, next_energy_bin])
        self.q[price_bin, energy_bin, action] = (1 - self.alpha) * current + self.alpha * target

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
        return self
