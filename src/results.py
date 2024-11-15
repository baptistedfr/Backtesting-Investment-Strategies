from dataclasses import dataclass
from typing import Union
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
        strategy_name (str) : name of the used strategy
        ptf_weights (optional pd.DataFrame) : weights of every asset over time
    """

    """---------------------------------------------------------------------------------------
    -                                 Class arguments                                        -
    ---------------------------------------------------------------------------------------"""

    ptf_values : pd.Series
    strategy_name : str
    ptf_weights : Optional[pd.DataFrame] = None

    df_statistics : pd.DataFrame = None
    ptf_value_plot : go.Figure = None
    ptf_weights_plot : Union[go.Figure, list[go.Figure]] = None

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
            self.strategy_name: [self.annualized_return, self.annualized_vol, self.sharpe_ratio, self.sortino_ratio, self.max_drawdown]
        }
        df = pd.DataFrame(data)
        df[self.strategy_name] = df[self.strategy_name].map(lambda x: "{:.2%}".format(x))

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
        if isinstance(all(dates), str):
            dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
        
        if self.strategy_name == "Benchmark":
            fig = go.Figure(data=go.Scatter(x=dates, y=strat_values,name=self.strategy_name, line=dict(dash='dot')))
        else : 
            fig = go.Figure(data=go.Scatter(x=dates, y=strat_values,name=self.strategy_name))
        fig.update_layout(title='Strategy performance', xaxis_title='Dates', yaxis_title='Portfolio Values', 
                          font=dict(family="Courier New, monospace", size=14,color="RebeccaPurple"))
        
        self.ptf_value_plot = fig
    
    def weights_plot(self):
        if self.ptf_weights is not None :
            fig = go.Figure()
            '''Attribut "stackgroup" permet d'empiler les plots'''
            for column in self.ptf_weights.columns:
                fig.add_trace(go.Scatter(x=self.ptf_weights.index, y=self.ptf_weights[column],
                                         mode='lines', stackgroup='one', name=column))

            fig.update_layout(title=f"Evolution of portfolio weights for strategy : {self.strategy_name}", xaxis_title="Date",
                yaxis_title="Weight (%)", yaxis=dict(tickformat=".0%", range=[0, 1]),
                legend_title="Assets", hovermode="x unified")
            
            self.ptf_weights_plot = fig

    """---------------------------------------------------------------------------------------
    -                                 Results Comparison                                     -
    ---------------------------------------------------------------------------------------"""

    @staticmethod
    def compare_results(results : list["Results"]) -> "Results":
        """
        Compare multiple strategy results and returns a new Results instance.
        If the strategy is compared with the benchmark, the comparaison result is named after the strategy to anable comparaison with other strategies afterward.
        
        Args:
            results (list[Results]) = list of different strategy results
        
        Returns:
            Results: A new Results object containing the combined statistics and comparison plot.
        """

        combined_statistics = Results.combine_df(results)
        fig = Results.combine_value_plot(results)
        weight_plots = Results.combine_weight_plot(results)

        if "Benchmark" in [res.strategy_name for res in results]:
            strat_name = [res.strategy_name for res in results if res.strategy_name != "Benchmark"][0]
        else :
            strat_name = "Comparaison"

        return Results(ptf_values=results[0].ptf_values, ptf_weights=results[0].ptf_weights,
                       df_statistics=combined_statistics, ptf_value_plot=fig,
                       ptf_weights_plot=weight_plots, strategy_name=strat_name)
    
    @staticmethod
    def combine_df(results : list["Results"]) -> go.Figure:
        """
        Combine the statistics dataframes and filter the columns to avoid having the benchmark twice

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of strategy values over time
        """

        '''Combine the statistics DataFrames'''
        combined_statistics = pd.DataFrame(columns=["Metrics"])
        is_benchmark= False
        for result in results:
            df_stats = result.df_statistics.copy()
            
            '''Get only one backtest column'''
            if "Benchmark" in df_stats.columns and is_benchmark is False:
                is_benchmark = True
            elif "Benchmark" in df_stats.columns and is_benchmark is True:
                df_stats = df_stats.drop('Benchmark', axis=1)

            combined_statistics = pd.merge(
                combined_statistics, df_stats, on="Metrics", how="outer"
            )
        '''Reorganise the columns'''
        cols = [col for col in combined_statistics.columns if col != "Benchmark"]
        if is_benchmark:
            cols+=["Benchmark"]
        combined_statistics = combined_statistics[cols]

        return combined_statistics
    
    @staticmethod
    def combine_value_plot(results : list["Results"]) -> go.Figure:
        """
        Combine the value plots by adding each trace or each plot results in a new plotly figure.
        To avoid adding twice the benchmark plot we check if the benchmark as already been added in the fig by check the scatter names.

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of strategy values over time
        """
        fig = go.Figure()
        for result in results:
            for scatter in result.ptf_value_plot.data:
                if scatter.name != "Benchmark" or (scatter.name == "Benchmark" and "Benchmark" not in [scatter.name for scatter in fig.data]):
                    fig.add_trace(scatter)

        fig.update_layout(title="Multiple Strategies Performance Comparison", xaxis_title="Date",
            yaxis_title="Portfolio Value", font=dict(family="Courier New, monospace", size=14, color="RebeccaPurple"))
        
        return fig
    
    @staticmethod
    def combine_weight_plot(results : list["Results"]) -> go.Figure:
        """
        Combine the weights plots.
        As the function is also used to compare with the benchmark, we want to return only one weight plot if it's the case else we return both.

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of strategy weights over time
        """

        if "Benchmark" not in [res.strategy_name for res in results]:
            '''In this case, we can store all weights evolution in a list of figures'''
            weight_plots = []
            for result in results:
                if result.ptf_weights_plot is not None:
                    weight_plots.append(result.ptf_weights_plot)
        else :
            '''Return only the weigt plot of the strategy and not the benchmark'''
            strat_name = [res.strategy_name for res in results if res.strategy_name != "Benchmark"][0]
            weight_plots = [res.ptf_weights_plot for res in results if res.strategy_name == strat_name][0]

        return weight_plots