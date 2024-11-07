from dataclasses import dataclass
from datetime import datetime
import plotly.graph_objects as go
from typing import Optional
import pandas as pd
import numpy as np

@dataclass
class Results:
    """
    Output from the backtest : computed statistic from the portfolio values and create main plots

    Args: 
        ptf_values (pd.Series) : value of the strategy over time
        ptf_weights (optional pd.DataFrame) : weights of every asset over time
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    ptf_values : pd.Series
    ptf_weights : Optional[pd.DataFrame] = None

    df_statistics : pd.DataFrame = None
    ptf_value_plot : go.Figure = None
    ptf_weights_plot : go.Figure = None
    other_weights_plot : go.Figure = None

    """---------------------------------------------------------------------------------------
    -                                 Generate Statistics                                    -
    ---------------------------------------------------------------------------------------"""

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

    def get_statistics(self) -> pd.DataFrame:
        data = {
            "Metrics": ["Annualized Return", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawn-Down"],
            "Values": [self.annualized_return, self.annualized_vol, self.sharpe_ratio, self.sortino_ratio, self.max_drawdown]
        }
        df = pd.DataFrame(data)
        df["Values"] = df["Values"].map(lambda x: "{:.2%}".format(x))

        self.df_statistics = df

    """---------------------------------------------------------------------------------------
    -                                   Generate Plots                                       -
    ---------------------------------------------------------------------------------------"""

    def create_plots(self) :
        self.strat_plot()
        self.weights_plot()

    def strat_plot(self) :

        strat_values = list(self.ptf_values)
        dates = list(self.ptf_values.index)
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
        self.ptf_value_plot = fig
    
    def weights_plot(self):
        if self.ptf_weights is not None :
            fig = go.Figure()
            '''Attribut "stackgroup" permet d'empiler les plots'''
            for column in self.ptf_weights.columns:
                fig.add_trace(go.Scatter(
                    x=self.ptf_weights.index, 
                    y=self.ptf_weights[column],
                    mode='lines',
                    stackgroup='one',
                    name='Strategy'
                ))

            fig.update_layout(
                title="Evolution of portfolio weights",
                xaxis_title="Date",
                yaxis_title="Weight (%)",
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                legend_title="Actifs",
                hovermode="x unified"
            )
            self.ptf_weights_plot = fig

    """---------------------------------------------------------------------------------------
    -                                 Results Comparison                                     -
    ---------------------------------------------------------------------------------------"""

    def compare_with(self, other: "Results", name_self: str = "Strategy", name_other: str = "Benchmark") -> "Results":
        """Compare this Result instance with another Result instance : comparison of statistics and plot
        
        Args:
            other (Results): The other Results object to compare with
            name_self (str): The label of the current strategy
            name_other (str): The label of the comparison strategy
        
        Returns:
            Results: A new Results object containing the combined statistics and comparison plot.
        """
        
        '''Combine the two statistics DataFrames'''
        df_self = self.df_statistics.copy()
        df_self.rename(columns={"Values": name_self}, inplace=True)
        
        df_other = other.df_statistics.copy()
        df_other.rename(columns={"Values": name_other}, inplace=True)
        
        df_comparison = pd.merge(df_self, df_other, on="Metrics", how="outer")
        
        '''Combine the two plots'''
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.ptf_values.index,
            y=self.ptf_values,
            mode='lines',
            name=name_self
        ))
        
        fig.add_trace(go.Scatter(
            x=other.ptf_values.index,
            y=other.ptf_values,
            mode='lines',
            name=name_other,
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title="Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            font=dict(family="Courier New, monospace", size=14, color="RebeccaPurple")
        )

        '''Returns a new instance of Results with new statistics DataFrame and plot'''
        if name_other == "Benchmark":
            new_output = Results(ptf_values=self.ptf_values,
                                ptf_weights=self.ptf_weights,
                                df_statistics=df_comparison,
                                ptf_value_plot=fig, 
                                ptf_weights_plot=self.ptf_weights_plot)
        else :
            new_output = Results(ptf_values=self.ptf_values,
                                ptf_weights=self.ptf_weights,
                                df_statistics=df_comparison,
                                ptf_value_plot=fig, 
                                ptf_weights_plot=self.ptf_weights_plot,
                                other_weights_plot=other.ptf_weights_plot)
        
        return new_output