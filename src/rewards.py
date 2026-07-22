"""Reward 1 and Reward 2 (paper Section III-C).

Reward 1 is the naive instant profit/cost of the charge/discharge decision.
Reward 2 subtracts/adds a moving-average price instead of the raw price, so
charging below the recent average is rewarded (not just charging cheaply in
absolute terms) -- this is what lets the agent explore arbitrage opportunities
instead of collapsing to "never charge" (paper Fig. 4/5, Reward 1 fails to
turn a profit on real ISO-NE data while Reward 2 does).

Note: the paper's AMP objective (Sec. II) includes charge/discharge
efficiencies eta_c, eta_d (eta_d*d - (1/eta_c)*c), but the Reward 1 / Reward 2
formulas as printed in Sec. III-C omit them entirely (reward is just
+/- price * rate). We implement the literal Sec. III-C formulas by default
(efficiency_aware=False); pass efficiency_aware=True to fold eta_c/eta_d into
the reward the way the AMP objective does, which is the more physically
correct choice if your battery's round-trip efficiency is meaningfully below
100%.
"""


class MovingAveragePrice:
    """p_bar_t = (1 - eta) * p_bar_{t-1} + eta * p_t (Eq. 6). `eta` here is the
    paper's smoothing parameter, unrelated to the battery efficiencies also
    called eta_c/eta_d elsewhere -- unfortunate but literal notation clash in
    the paper itself.
    """

    def __init__(self, smoothing=0.1, initial_price=None):
        self.smoothing = smoothing
        self.value = initial_price

    def update(self, price):
        if self.value is None:
            self.value = price
        else:
            self.value = (1 - self.smoothing) * self.value + self.smoothing * price
        return self.value


def reward_1(price, c, d, eta_c=1.0, eta_d=1.0, efficiency_aware=False):
    if efficiency_aware:
        if c > 0:
            return -price * c / eta_c
        if d > 0:
            return price * d * eta_d
        return 0.0
    if c > 0:
        return -price * c
    if d > 0:
        return price * d
    return 0.0


def reward_2(price, avg_price, c, d, eta_c=1.0, eta_d=1.0, efficiency_aware=False):
    if efficiency_aware:
        if c > 0:
            return (avg_price - price) * c / eta_c
        if d > 0:
            return (price - avg_price) * d * eta_d
        return 0.0
    if c > 0:
        return (avg_price - price) * c
    if d > 0:
        return (price - avg_price) * d
    return 0.0
