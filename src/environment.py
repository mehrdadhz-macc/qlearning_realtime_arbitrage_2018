"""Storage arbitrage MDP (paper Sections II-III).

Implements the Arbitrage Maximization Problem (AMP, Eq. in Sec. II) and its
Lemma 1 result: the optimal charge/discharge policy is bang-bang, so the
action space collapses to exactly three choices -- charge at the maximum
feasible rate, hold, or discharge at the maximum feasible rate. This is what
makes tabular Q-learning over a small (price bin, energy bin) state space
tractable.
"""

import numpy as np


class StorageArbitrageEnv:
    """One episode == one pass through a price series, hour by hour.

    State: (price_bin, energy_bin), both discretized (Sec. III-A).
    Action: 0 = hold, 1 = charge at max feasible rate, 2 = discharge at max
    feasible rate (Sec. III-B, action set A = {-D~max, 0, C~max}, reordered
    here so HOLD is index 0). That ordering matters for more than labeling:
    numpy's argmax breaks ties by returning the FIRST maximal index, and a
    freshly-initialized or never-visited (price_bin, energy_bin) row is all
    zeros -- an exact tie. With HOLD at index 0, an untrained/tied state
    defaults to the harmless no-op. The alternative (discharge or charge
    first) is actively dangerous at the SoC boundary: starting from
    energy=e_min, an untrained greedy rollout would keep selecting
    "discharge" (infeasible at empty SoC, silently falls back to hold), so
    the battery would never charge in the first place and would sit stuck at
    zero profit for the whole episode -- exactly the failure mode this
    ordering avoids.
    """

    HOLD, CHARGE, DISCHARGE = 0, 1, 2

    def __init__(
        self,
        prices,
        capacity_mwh=8.0,
        max_rate_mw=1.0,
        e_min=0.0,
        efficiency_charge=1.0,
        efficiency_discharge=1.0,
        n_price_bins=10,
        price_bin_edges=None,
        initial_energy=0.0,
    ):
        self.prices = np.asarray(prices, dtype=float)
        self.T = len(self.prices)
        self.e_min = e_min
        self.e_max = capacity_mwh
        self.c_max = max_rate_mw
        self.d_max = max_rate_mw
        self.eta_c = efficiency_charge
        self.eta_d = efficiency_discharge
        self.initial_energy = initial_energy

        # Energy is always exactly a multiple of max_rate_mw away from
        # initial_energy under the bang-bang policy (Lemma 1), so an integer
        # grid at that resolution represents every reachable state exactly
        # -- no discretization error on the energy axis.
        self.n_energy_bins = int(round((self.e_max - self.e_min) / self.max_rate_mw_gcd())) + 1

        if price_bin_edges is not None:
            self.price_bin_edges = np.asarray(price_bin_edges, dtype=float)
        else:
            self.price_bin_edges = np.linspace(self.prices.min(), self.prices.max(), n_price_bins + 1)
        self.n_price_bins = len(self.price_bin_edges) - 1

        self.t = 0
        self.energy = initial_energy

    def max_rate_mw_gcd(self):
        # Charge and discharge rates are equal in this paper's case study;
        # kept as a separate method in case they ever diverge.
        return self.c_max

    def price_bin(self, price):
        idx = np.searchsorted(self.price_bin_edges, price, side="right") - 1
        return int(np.clip(idx, 0, self.n_price_bins - 1))

    def energy_bin(self, energy):
        idx = round((energy - self.e_min) / self.max_rate_mw_gcd())
        return int(np.clip(idx, 0, self.n_energy_bins - 1))

    def state(self):
        price = self.prices[self.t]
        return (self.price_bin(price), self.energy_bin(self.energy))

    def reset(self):
        self.t = 0
        self.energy = self.initial_energy
        return self.state()

    def feasible_rates(self):
        """(C~max, D~max): max feasible charge / discharge given current SoC headroom."""
        c_tilde = min(self.c_max, self.e_max - self.energy)
        d_tilde = min(self.d_max, self.energy - self.e_min)
        return max(c_tilde, 0.0), max(d_tilde, 0.0)

    def step(self, action):
        """Advance one hour. Returns (next_state, price, action_taken, c, d, done).

        Reward is intentionally NOT computed here -- src/rewards.py takes
        (price, c, d, c_tilde, d_tilde) so Reward 1 and Reward 2 can be swapped
        without touching environment dynamics.
        """
        price = self.prices[self.t]
        c_tilde, d_tilde = self.feasible_rates()

        c = d = 0.0
        if action == self.CHARGE and c_tilde > 0:
            c = c_tilde
        elif action == self.DISCHARGE and d_tilde > 0:
            d = d_tilde
        # else HOLD, or the requested action had no feasible headroom -> hold

        self.energy = self.energy + c - d
        self.t += 1
        done = self.t >= self.T
        next_state = self.state() if not done else None
        return next_state, price, c, d, c_tilde, d_tilde, done
