"""Shared helper for the single-seed comparison plots (best-seed-per-reward,
best-held-out-seed-per-reward, best-seed-reward-comparison): a two-panel
figure with the raw price series on top and cumulative-profit curves below,
sharing the same hour x-axis. A dual-axis overlay (price and profit on the
same axes with two y-scales) is a known anti-pattern -- misleading and hard
to read -- so this stacks two single-axis panels instead, letting a reader
see the actual price driving any jump in the profit curve rather than
inferring it from prose.
"""

import matplotlib.pyplot as plt
import numpy as np


def money(x):
    """Format a dollar amount for matplotlib text with the $ escaped --
    matplotlib's mathtext treats any two literal '$' in one string as a
    math-mode span (silently eating spaces and the delimiters themselves),
    so any text with more than one dollar amount (e.g. a legend entry with
    both an online and a greedy-after-training figure) breaks unless every
    '$' is escaped as '\\$'."""
    return f"\\${x:,.2f}"


def make_price_profit_figure(prices, curves, colors, profit_title, price_label="Price ($/MWh)"):
    """curves: dict of series label -> (color_key, array) or (color_key, array,
    legend_suffix), array same length as prices. Final cumulative profit is
    appended to each legend entry AND annotated directly at the end of its
    curve, so the headline number is readable from the plot itself, not
    just cross-referenced against a caption. legend_suffix (optional,
    3-tuple form) appends extra text to the legend entry only -- e.g. the
    greedy (post-training, epsilon=0) profit alongside the epsilon=0.9
    online curve's own endpoint, since epsilon never decays during training
    (Algorithm 1) so the online curve's endpoint is NOT a clean read of
    what the learned policy alone actually achieved."""
    fig, (price_ax, profit_ax) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1, 2])

    hours = np.arange(len(prices))
    price_ax.plot(hours, prices, color="tab:gray", linewidth=0.5)
    price_ax.set_ylabel(price_label)
    price_ax.set_title("Underlying price series", fontsize=10, color="dimgray", loc="left")

    for label, spec in curves.items():
        color_key, curve = spec[0], spec[1]
        legend_suffix = spec[2] if len(spec) > 2 else ""
        color = colors[color_key]
        final_value = curve[-1]
        profit_ax.plot(np.arange(len(curve)), curve,
                       label=f"{label}: {money(final_value)}{legend_suffix}", color=color, linewidth=1)
        profit_ax.annotate(money(final_value), (len(curve) - 1, final_value), color=color,
                           fontsize=9, fontweight="bold", xytext=(6, 0),
                           textcoords="offset points", va="center")
    profit_ax.axhline(0, color="gray", linewidth=0.7)
    profit_ax.set_xlabel("Time (hour)")
    profit_ax.set_ylabel("Cumulative profit ($)")
    profit_ax.set_title(profit_title, fontsize=11, loc="left")
    profit_ax.legend()
    profit_ax.margins(x=0.09)  # room for the end-of-curve annotations

    fig.tight_layout()
    return fig
