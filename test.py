from src.data_input import DataInput, InputType, FrequencyType, Index

data_equity = DataInput(data_type=InputType.EQUITY,
                        tickers=['MC.PA', 'OR.PA'],
                        start_date='2023-10-01',
                        end_date='2024-10-01',
                        frequency=FrequencyType.DAILY)

data_crypto = DataInput(data_type=InputType.CRYPTO,
                        tickers=['ETHUSDT', 'BTCUSDT', 'SOLUSDT'],
                        frequency=FrequencyType.DAILY)
                 
data_custom = DataInput(data_type=InputType.CUSTOM,
                        file_path='data/custom.xlsx')

data_index = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                        index=Index.CAC40,
                        start_date='2023-10-01',
                        end_date='2024-10-01',
                        frequency=FrequencyType.DAILY)

print("end")