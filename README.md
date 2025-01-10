# Backtester_POO_272_MCD

**Backtester_POO_272_MCD** est un package Python conçu pour effectuer des backtests efficaces et flexibles sur des stratégies quantitatives. 

Ce projet vise à fournir un cadre structurel orienté objet permettant d’évaluer des stratégies financières sur des données historiques, tout en facilitant l'analyse et l'adaptation des paramètres.

Un manuel détaillé est disponible dans le notebook `manual.ipynb`


### Installation avec pip

Pour installer ou mettre à jour backtester_poo_272_mcd via `pip`, exécutez la commande suivante :

```bash
pip install backtester_poo_272_mcd
```
### Exemple

```bash
'''Initialisation des données'''
data = DataInput(data_type=InputType.FROM_INDEX_COMPOSITION,
                index=Index.CAC40,
                start_date='2015-10-01',
                end_date='2024-10-01',
                frequency=FrequencyType.WEEKLY,
                benchmark=Benchmark.CAC40)
'''Initialisation du backtest'''
backtest = Backtester(data_input=data)
'''Initialisation des stratégies'''
strategy_mkw = OptimalSharpeStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1)
strat_kernel = KernelSkewStrategy(rebalance_frequency=FrequencyType.MONTHLY, lookback_period=1)
'''Run du backtest'''
results_mkw = backtest.run(strategy=strategy_mkw, initial_amount=1000.0, fees=0.0)
result_kernel = backtest.run(strategy=strat_kernel, initial_amount=1000.0, fees=0.0)
'''Visualisation des résultats'''
combined_results = Results.compare_results([results_mkw, result_kernel])
print(combined_results.df_statistics.head(10))
combined_results.ptf_value_plot.show()
combined_results.ptf_drawdown_plot.show()
for plot in combined_results.ptf_weights_plot:
    plot.show()
```
