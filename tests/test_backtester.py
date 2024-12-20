import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from my_package.tools import InputType, FrequencyType, Index, Benchmark
from my_package.strategy import *
from my_package import Backtester
from my_package import DataInput
from my_package import Results

@pytest.fixture
def backtester():
    # Données de test : Simuler des prix pour 3 actifs sur 5 périodes
    data = {
        'Date': pd.date_range(start="2023-01-01", periods=5, freq='D'),
        'Asset1': [100, 101, 102, 101, 103],
        'Asset2': [50, 51, 52, 51, 53],
        'Asset3': [200, 199, 198, 202, 204]
    }
    df_prices = pd.DataFrame(data)
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    data = DataInput(data_type=InputType.FROM_DATAFRAME, 
                           custom_df=df_prices, 
                           initial_weights=[0.4, 0.3, 0.3], 
                           frequency=FrequencyType.DAILY)
    return Backtester(data_input=data)

def test_df_prices(backtester : Backtester):
    """
    Vérifier que la propriété df_prices renvoie bien le DataFrame des prix et sous la bonne forme
    """
    df_prices = backtester.df_prices
    assert isinstance(df_prices, pd.DataFrame)
    assert df_prices.shape == (5, 4)  # Vérifier que la taille est correcte (5 périodes et 4 colonnes)
    assert "Date" in df_prices.columns  # Vérifier que la colonne "Date" est présente
    assert "Asset1" in df_prices.columns  # Vérifier la présence d'une colonne d'actifs

def test_dates(backtester : Backtester):
    """
    Vérifier que la propriété dates renvoie une liste de dates correctes
    """
    dates = backtester.dates
    assert isinstance(dates, pd.Series)
    assert len(dates) == 5  # Vérifier qu'il y a bien 5 dates
    assert all(isinstance(date, datetime) for date in dates)  # Vérifier que chaque élément est une instance datetime
    assert dates[0] == datetime(2023, 1, 1)  # Vérifier que la première date est correcte


def test_initial_weights(backtester : Backtester):
    """
    Vérifier que les poids initiaux sont correctement initialisés
    """
    weights = backtester.initial_weights_value
    assert np.allclose(weights, np.array([0.4, 0.3, 0.3]))

def test_backtest_length(backtester : Backtester):
    """
    Vérifier la longueur du backtest (doit être égale au nombre de périodes)
    """
    assert backtester.backtest_length == 5

def test_backtest_nb_assets(backtester : Backtester):
    """
    Vérifier que le nombre d'actifs est exact
    """
    assert backtester.nb_assets == 3

def test_run_backtest(backtester : Backtester):
    """
    Vérifier que la fonction run renvoie bien un objet de la classe result
    """
    strategy = EqualWeightStrategy(rebalance_frequency=FrequencyType.DAILY, lookback_period=0)
    results = backtester.run(strategy, initial_amount=1000.0)
    assert isinstance(results, Results)


def test_run_backtest_invalid_strategy(backtester : Backtester):
    """
    Tester un cas où la stratégie de rebalancement dépasse la fréquence des prix
    """
    strategy = EqualWeightStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1)  # Fréquence de rebalance trop haute
    with pytest.raises(ValueError):
        backtester.run(strategy, initial_amount=1000.0)


def test_run_backtest_no_data(backtester : Backtester):
    """
    Tester un cas où les données sont insuffisantes pour le backtest
    """
    strategy = EqualWeightStrategy(rebalance_frequency=FrequencyType.DAILY, lookback_period=1)
    with pytest.raises(ValueError):
        backtester.run(strategy, initial_amount=1000.0)