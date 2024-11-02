from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np

@dataclass
class Results:
    """
    Output from the backtest : computed statistic from the portfolio values and create main plots

    Args: 
        ptf_weights (pd.DataFrame) : weights of every asset over time
        ptf_values (pd.Series) : value of the strategy over time
    """

    ptf_weights : pd.DataFrame
    ptf_values : pd.Series

    @property
    def ptf_returns(self) -> list[float]:
        return list(pd.Series(self.ptf_values).pct_change().iloc[1:])
    
    @property
    def total_return(self) -> float:
        return (self.ptf_values.iloc[-1] / self.ptf_values.iloc[0]) - 1
    
    '''CAGR formula'''
    @property
    def annualized_return(self) -> float:
        return (self.ptf_values.iloc[-1]/self.ptf_values.iloc[0])**(252/len(self.ptf_values)) - 1

    @property
    def annualized_vol(self) -> float:
        return np.std(self.ptf_returns) * np.sqrt(252)

    @property
    def sharpe_ratio(self) -> float:
        return self.annualized_return/self.annualized_vol
    
    @property
    def sortino_ratio(self) -> float:
        downside_returns = [r for r in self.ptf_returns if r < 0]
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(len(self.ptf_returns))
        return self.annualized_return / downside_std
        
    @property
    def max_drawdown(self) -> float:
        cumulative_returns = np.cumsum(self.ptf_returns)
        previous_peaks = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - previous_peaks
        max_drawdown = np.min(drawdowns)
        return max_drawdown

    @property
    def statistics(self) -> pd.DataFrame:
        data = {
            "Metrics": ["Annualized Return", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawn-Down"],
            "Values": [self.annualized_return, self.annualized_vol, self.sharpe_ratio, self.sortino_ratio, self.max_drawdown]
        }
        df = pd.DataFrame(data)
        df["Values"] = df["Values"].map(lambda x: "{:.2%}".format(x))
        return df

    @property
    def strat_plot(self) -> go.Figure:

        strat_values = list(self.ptf_values)
        dates = list(self.ptf_weights.index)
        dates = [datetime.strftime(d, "%Y-%m-%d") for d in dates]
        
        fig = go.Figure(data=go.Scatter(x=dates, y=strat_values))
        fig.update_layout(
            title='Strategy performance',
            xaxis_title='Dates',
            yaxis_title='Portfolio Values',
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="RebeccaPurple"
            )
        )
        return fig