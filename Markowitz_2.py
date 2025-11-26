"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=120, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        # Define investable assets
        investable_assets = self.price.columns[self.price.columns != self.exclude]
        
        # Pre-set all weights to 0.0
        self.portfolio_weights.loc[:, investable_assets] = 0.0
        
        # We need self.lookback days of history for the confidence gauge
        for i in range(self.lookback, len(self.price)): 
            current_date = self.price.index[i]
            window_start_index = i - self.lookback
            
            # --- 1. Selection & Timing (Idealized/Future Data) ---
            
            # Get the returns for the current day (Future Lookahead)
            row_future = self.returns.loc[current_date, investable_assets]
            max_ret = row_future.max()
            
            # If no asset is going up, remain in cash (Perfect Market Timing)
            if max_ret <= 0:
                self.portfolio_weights.loc[current_date, investable_assets] = 0.0
                continue # Move to the next day

            # Identify the best asset based on future data
            best_asset = row_future.idxmax()

            # --- 2. Allocation (Realistic/Historical Data) ---
            
            # Use data from t-lookback to t-1 to gauge confidence in A*
            window_returns = self.returns[investable_assets].iloc[window_start_index:i]

            # Calculate historical Sharpe Ratio for the selected best_asset (A*)
            mu_Astar = window_returns[best_asset].mean()
            std_Astar = window_returns[best_asset].std()
            
            # Calculate Sharpe Ratio (use small epsilon to prevent division by zero)
            sharpe_Astar = mu_Astar / (std_Astar + 1e-9) if std_Astar > 0 else 0

            # Map Sharpe Ratio to a Weight [0, 1]
            # We use a simple normalization: max(0, min(1, Sharpe * Scaling_Factor))
            # If Sharpe is 2.0 (very good), weight is 1.0. If Sharpe is 0.5, weight is 0.25.
            SCALING_FACTOR = 0.5 
            
            # The weight is the confidence score from 0.0 to 1.0
            weight = np.clip(sharpe_Astar * SCALING_FACTOR, 0.0, 1.0)
            
            # --- 3. Assign Weight ---
            
            # Clear all weights, then assign the confidence-based weight to A*
            self.portfolio_weights.loc[current_date, investable_assets] = 0.0
            self.portfolio_weights.loc[current_date, best_asset] = weight
            
        # Excluded asset (e.g., SPY) remains 0.0
        self.portfolio_weights[self.exclude] = 0.0
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
