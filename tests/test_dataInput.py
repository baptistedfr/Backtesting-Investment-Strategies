import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from backtester_poo_272_mcd.tools import InputType, FrequencyType, Index, Benchmark
from backtester_poo_272_mcd import DataInput

def test_data_input_no_benchmark():
    """
    Tests des différents types de DataInputs sans benchmark
    """
    data_equity = DataInput(data_type=InputType.EQUITY,
                        tickers=['GLE.PA', 'OR.PA','MC.PA','VIV.PA','TTE.PA'],
                        start_date='2020-10-01',
                        end_date='2024-10-01',
                        frequency=FrequencyType.WEEKLY)
    
    data_crypto = DataInput(data_type=InputType.CRYPTO,
                        tickers=['BTCUSDT','ETHUSDT','PEPEUSDT','DOGEUSDT','SOLUSDT'],
                        start_date='2018-10-01',
                        end_date='2024-11-15',
                        frequency=FrequencyType.WEEKLY)
    
    data_index = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                index=Index.CAC40,
                start_date='2010-10-01',
                end_date='2024-10-01',
                frequency=FrequencyType.WEEKLY)

    df = {
        'Date': pd.date_range(start="2023-01-01", periods=5, freq='D'),
        'Asset1': [100, 101, 102, 101, 103],
        'Asset2': [50, 51, 52, 51, 53],
        'Asset3': [200, 199, 198, 202, 204]
    }
    df_prices = pd.DataFrame(df)
    data_df = DataInput(data_type=InputType.FROM_DATAFRAME,
                custom_df=df_prices,
                frequency=FrequencyType.DAILY)
    
    datas_inputs = [data_equity, data_crypto, data_index ,data_df]
    for input in datas_inputs:
        assert isinstance(input.df_prices, pd.DataFrame)
        assert "Date" in input.df_prices.columns  # Vérifier que la colonne "Date" est présente
        assert len(input.df_prices) > 1# Verifier que le dataset comprend des données

def test_data_input_benchmark():
    """
    Tests des différents types de DataInputs sans benchmark
    """
    data_equity = DataInput(data_type=InputType.EQUITY,
                        tickers=['GLE.PA', 'OR.PA','MC.PA','VIV.PA','TTE.PA'],
                        start_date='2020-10-01',
                        end_date='2024-10-01',
                        frequency=FrequencyType.WEEKLY,
                        benchmark=Benchmark.CAC40)
    
    data_crypto = DataInput(data_type=InputType.CRYPTO,
                        tickers=['BTCUSDT','ETHUSDT','PEPEUSDT','DOGEUSDT','SOLUSDT'],
                        start_date='2018-10-01',
                        end_date='2024-11-15',
                        frequency=FrequencyType.WEEKLY,
                        benchmark=Benchmark.BTC)
    
    data_index = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                index=Index.CAC40,
                start_date='2010-10-01',
                end_date='2024-10-01',
                frequency=FrequencyType.WEEKLY,
                benchmark=Benchmark.CAC40)

    df = {
        'Date': pd.date_range(start="2023-01-02", periods=5, freq='D'),
        'Asset1': [100, 101, 102, 101, 103],
        'Asset2': [50, 51, 52, 51, 53],
        'Asset3': [200, 199, 198, 202, 204]
    }
    df_prices = pd.DataFrame(df)
    data_df = DataInput(data_type=InputType.FROM_DATAFRAME,
                custom_df=df_prices,
                frequency=FrequencyType.DAILY,
                benchmark=Benchmark.CAC40)
    
    datas_inputs = [data_equity,data_crypto,data_index, data_df]
    
    for input in datas_inputs:
        assert isinstance(input.df_prices, pd.DataFrame)
        assert "Date" in input.df_prices.columns  # Vérifier que la colonne "Date" est présente
        assert len(input.df_prices)> 1# Verifier que le dataset comprend des données
        assert len(input.df_prices) == len(input.df_benchmark) #Verifier que la taille du benchmark est identique avec celle des prix
        assert input.df_benchmark is not None # Verifier que le bench n est pas nul
        assert 'Date' in input.df_benchmark.columns