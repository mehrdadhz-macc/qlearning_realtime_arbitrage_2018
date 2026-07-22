"""Reproduce paper Fig. 1 style: raw real-time price + moving-average price overlay."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_price_series
from src.rewards import MovingAveragePrice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train/isone_rt_hourly_lmp_2016.csv")
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    timestamps, prices = load_price_series(args.data)
    avg = MovingAveragePrice(smoothing=args.smoothing)
    avg_series = [avg.update(p) for p in prices]

    plt.figure(figsize=(10, 4))
    plt.plot(prices, label="Real-time price", linewidth=0.7)
    plt.plot(avg_series, label="Average price", linewidth=1.2)
    plt.xlabel("Time (hour)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"ISO-NE real-time hourly price ({Path(args.data).stem})")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out) if args.out else Path("outputs/data_plots") / f"{Path(args.data).stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
