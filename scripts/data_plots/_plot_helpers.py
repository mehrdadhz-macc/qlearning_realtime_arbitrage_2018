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


def make_price_profit_figure(prices, curves, colors, profit_title, price_label="Price ($/MWh)"):
    """curves: dict of series label -> (color_key, array), array same length as prices.
    Final cumulative profit is appended to each legend entry AND annotated
    directly at the end of its curve, so the headline number is readable
    from the plot itself, not just cross-referenced against a caption."""
    fig, (price_ax, profit_ax) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1, 2])

    hours = np.arange(len(prices))
    price_ax.plot(hours, prices, color="tab:gray", linewidth=0.5)
    price_ax.set_ylabel(price_label)
    price_ax.set_title("Underlying price series", fontsize=10, color="dimgray", loc="left")

    for label, (color_key, curve) in curves.items():
        color = colors[color_key]
        final_value = curve[-1]
        profit_ax.plot(np.arange(len(curve)), curve,
                       label=f"{label}: ${final_value:,.2f}", color=color, linewidth=1)
        profit_ax.annotate(f"${final_value:,.2f}", (len(curve) - 1, final_value), color=color,
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
