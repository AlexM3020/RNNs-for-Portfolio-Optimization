"""Explicit commands for archive reporting, downloads, and fresh experiments."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM portfolio-allocation course replication")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="Render recorded metrics without data downloads or training")
    report.add_argument("--root", type=Path, default=Path.cwd())
    report.add_argument("--output", type=Path, default=Path("runs/historical-report"))
    download = commands.add_parser("download", help="Explicit Yahoo Finance data download")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--start", default="2010-01-01")
    download.add_argument("--end", default="2025-01-01")
    run = commands.add_parser("backtest", help="Run a new expanding-window experiment")
    run.add_argument("--prices", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    for name, default in [("lookback", 50), ("hidden-size", 128), ("epochs", 100),
                          ("batch-size", 64), ("patience", 20), ("seed", 42),
                          ("first-test-year", 2012), ("step-years", 2), ("threads", 2)]:
        run.add_argument(f"--{name}", type=int, default=default)
    run.add_argument("--learning-rate", type=float, default=0.0001)
    run.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    try:
        if args.command == "report":
            from .reporting import render_historical
            render_historical(args.root, args.output)
            print(f"Recorded metrics rendered in {args.output}; no training or downloads.")
        elif args.command == "download":
            from .data import download_prices
            download_prices(args.output, args.start, args.end)
            print(f"Downloaded prices to {args.output}")
        else:
            import torch
            from .backtest import Settings, run_backtest
            from .reporting import plot_run
            if args.threads < 1:
                raise ValueError("Threads must be positive.")
            torch.set_num_threads(args.threads)
            settings = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
            summary = run_backtest(args.prices, args.output, settings)
            plot_run(args.output)
            print(json.dumps(summary, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
