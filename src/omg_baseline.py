"""Online Modified Greedy (OMG) baseline -- Qin, Chow, Yang, Rajagopal (2016),
"Online Modified Greedy Algorithm for Storage Control under Uncertainty,"
arXiv:1405.7789 (published IEEE Trans. Power Systems 31(3), 2016). This is
the baseline Wang & Zhang's Sec. IV-C compares their RL policy against
([15] in their references).

Read directly from the source paper rather than guessed at from Wang &
Zhang's one-paragraph secondary description (a thresholding strategy that
"needs to estimate the bounds of prices and assume storages are big
enough"). For the pure arbitrage cost function (their Example 4) with
identity charge/discharge conversion (their Example 8(i) -- the paper's own
closed-form solution, algebraically simplest case), OMG reduces to exactly
that: a threshold on the *previous* storage level that depends on the
*current* price.

    threshold_t = -Gamma - (W/lambda) * p_t
    u*_t = discharge at max rate   if E_{t-1} > threshold_t
         = charge at max rate      otherwise

The paper's own Example 8(i) prints this as "s_t > (W*p_t/lambda) - Gamma"
(positive p_t coefficient). Extracting that from the PDF and testing it
numerically (e_max=8MWh, price range [-50,150]) showed it backwards: the
threshold hits $10 at the $150 high-price end -- above the $8 max SoC, so
the battery would NEVER discharge even at peak price. That's almost
certainly an OCR/extraction sign error, not a paper error. Re-derived
independently three ways, all agreeing on the sign used here instead:
(1) directly minimizing the online step's objective (17) -- a linear
function of u_t with coefficient lambda*(s_t+Gamma) + W*p_t; solving
"coefficient >= 0 -> discharge" for s_t gives exactly this threshold;
(2) the paper's own Lemma 3 (Appendix A) states the discharge condition as
lambda*(s_t+Gamma) + W*Dg_bar >= 0 -- identical structure, subgradient
bound in place of the realized price; (3) numerically, this version puts
the discharge threshold near S_min at the highest price and near S_max at
the lowest price (the correct arbitrage direction), the opposite of the
as-printed version. Gamma itself (below) was cross-checked against the
paper's Eq. 13/14/16 three independent ways and needed no correction --
only Example 8(i)'s printed price-term sign was wrong.

lambda = 1 here always (no self-discharge/dissipation term) -- Wang &
Zhang's own storage model (Sec. II, Eq. 1: E_t = E_{t-1} + c_t - d_t) has
none, so using lambda=1 keeps this baseline on the identical storage
dynamics as the RL policy it's compared against, not a generalization Wang
& Zhang's AMP doesn't have. One convenient consequence (paper's Remark 3):
for lambda=1, OMG's two possible offline parameter-selection strategies
(the "greedy" maxW closed form, and minS's semidefinite program) coincide
-- so there's a single well-defined (Gamma, W), not an implementation
choice to make. Only maxW needs to be computed; minS's SDP (Lemma 1) isn't
needed at all for this case.

Identity conversion (eta_c = eta_d = 1) matches this project's own
existing default for efficiency (train.py/evaluate.py's
--efficiency-charge/--efficiency-discharge default to 1.0), for the same
reason documented there: neither Wang & Zhang's Sec. III-C reward formulas
nor their Sec. IV numerical setup ever give concrete non-1.0 efficiency
values to match.

The only estimation OMG needs is [p_min, p_max] (Qin et al.'s Dg/Dg-bar
for the arbitrage cost, their Example 7(i): the subgradient of a linear
price*power cost is just the price itself, so its bounds are literally
the price bounds). Estimated causally from a prefix of the training
series here (`fit_omg_parameters`'s `calibration_prices` argument),
mirroring this project's own `--bin-calibration-hours` treatment of the
Q-learning agent's price bins -- same no-lookahead principle, applied to
this baseline's own only free parameter.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OMGParameters:
    """Offline-phase output (Qin et al. Sec. III-A): the (Gamma, W) pair,
    plus the inputs used to derive them, kept for transparency/debugging."""
    gamma_shift: float
    weight: float
    p_min: float
    p_max: float
    e_min: float
    e_max: float
    c_max: float
    d_max: float


def fit_omg_parameters(calibration_prices, e_min, e_max, c_max, d_max):
    """Offline phase (Qin et al. Sec. III-A, 'maxW' approach, Eq. 16):
    estimate [p_min, p_max] from a (causal) price sample, then compute the
    unique (Gamma, W) for lambda=1 (Remark 3: maxW and minS coincide).

    W = [(e_max - e_min) - (c_max + d_max)] / (p_max - p_min)      (Eq. 15)
    Gamma = [p_min*(e_min + d_max) - p_max*(e_max - c_max)]
            / (p_max - p_min)                                      (Eq. 16, lambda=1)
    """
    calibration_prices = np.asarray(calibration_prices, dtype=float)
    p_min, p_max = float(calibration_prices.min()), float(calibration_prices.max())
    if p_max <= p_min:
        raise ValueError(f"Degenerate price range in calibration window: [{p_min}, {p_max}]")

    price_range = p_max - p_min
    weight = ((e_max - e_min) - (c_max + d_max)) / price_range
    if weight <= 0:
        raise ValueError(
            f"W_max={weight:.6g} <= 0: storage capacity ({e_max - e_min}) isn't large enough "
            f"relative to the charge+discharge rate ({c_max + d_max}) for OMG's feasibility "
            f"condition (Qin et al. Assumption A3 / Eq. 15's Wmax > 0 requirement)."
        )
    gamma_shift = (p_min * (e_min + d_max) - p_max * (e_max - c_max)) / price_range

    return OMGParameters(gamma_shift=gamma_shift, weight=weight, p_min=p_min, p_max=p_max,
                          e_min=e_min, e_max=e_max, c_max=c_max, d_max=d_max)


def omg_threshold(price, params):
    """-Gamma - W*p_t (lambda=1) -- see module docstring for why this is
    sign-corrected from Example 8(i)'s as-printed "(W*p_t/lambda) - Gamma"."""
    return -params.gamma_shift - params.weight * price


def select_omg_action(energy, price, params, hold_action, charge_action, discharge_action):
    """Online phase (Qin et al. Sec. III-A, Eq. 17, closed form Example
    8(i)): compare the storage level *before* this step's action against
    the price-dependent threshold. Always bang-bang for pure arbitrage --
    the arbitrage cost's subgradient w.r.t. u is the single point {p_t},
    so the online LP (17) always sits at a vertex of [U_min, U_max], never
    an interior point -- there is no "hold" region, matching Wang &
    Zhang's own Lemma 1 (bang-bang optimality) though for a different
    algebraic reason (linearity of Example 4's cost, vs. Lemma 1's LP
    argument for AMP directly). hold_action is accepted only so this
    function has the same shape as a Q-table action-selector and is never
    actually returned.
    """
    del hold_action  # never selected -- see docstring
    threshold = omg_threshold(price, params)
    return discharge_action if energy > threshold else charge_action
