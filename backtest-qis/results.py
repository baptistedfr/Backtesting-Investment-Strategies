from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

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

    def create_plot(self):

        strat_values = list(self.ptf_values)
        dates = list(self.ptf_weights.index)
        dates = [datetime.strftime(d, "%Y-%m-%d") for d in dates]
        
        fig = go.Figure(data=go.Scatter(x=dates, y=strat_values))
        fig.update_layout(
            title='Strategy performance',
            xaxis_title='Dates',
            yaxis_title='Ptf Values',
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="RebeccaPurple"
            )
        )
        self.strat_plot = fig