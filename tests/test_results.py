import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from backtester_poo_272_mcd.results import Results
from backtester_poo_272_mcd.tools import FrequencyType
import plotly.graph_objects as go

@pytest.fixture
def sample_results():
    # Exemple de données fictives pour tester les statistiques
    ptf_values = pd.Series([100, 110, 120, 115, 125], index=pd.date_range('2023-01-01', periods=5, freq='D'))
    ptf_weights = pd.DataFrame({
        'Asset1': [0.5, 0.4, 0.45, 0.5, 0.55],
        'Asset2': [0.5, 0.6, 0.55, 0.5, 0.45],
    }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
    return Results(ptf_values=ptf_values, 
                   strategy_name="Test Strategy", 
                   data_frequency=FrequencyType.DAILY, 
                   ptf_weights=ptf_weights)

def test_results_initialization(sample_results : Results):
    # Vérifie que l'objet est bien initialisé avec les bons attributs
    assert sample_results.strategy_name == "Test Strategy"
    assert isinstance(sample_results.ptf_values, pd.Series)
    assert isinstance(sample_results.ptf_weights, pd.DataFrame)

def test_total_return(sample_results : Results):
    # Vérifier si le retour total est calculé correctement
    expected_total_return = (125 / 100) - 1  # (dernière valeur / première valeur) - 1
    assert np.isclose(sample_results.total_return, expected_total_return, atol=1e-6)

def test_annualized_return(sample_results : Results):
    # Le calcul du rendement annualisé
    expected_annualized_return = (125 / 100) ** (FrequencyType.DAILY.value / len(sample_results.ptf_values)) - 1
    assert np.isclose(sample_results.annualized_return, expected_annualized_return, atol=1e-6)

def test_sharpe_ratio(sample_results : Results):
    # Calcul du Sharpe Ratio
    annualized_vol = sample_results.annualized_vol
    annualized_return = sample_results.annualized_return
    expected_sharpe_ratio = (annualized_return - 0.02) / annualized_vol  # rf=0.02
    assert np.isclose(sample_results.sharpe_ratio, expected_sharpe_ratio, atol=1e-6)

def test_compute_var(sample_results : Results):
    var_95 = sample_results.compute_VaR(alpha=0.95)
    assert isinstance(var_95, float)  # Devrait retourner un float
    assert var_95 < 0  # La VaR devrait être négative, car elle représente une perte potentielle

def test_compute_cvar(sample_results : Results):
    cvar_95 = sample_results.compute_CVaR(alpha=0.95)
    assert isinstance(cvar_95, float)  # Devrait retourner un float
    assert cvar_95 < 0  # Comme pour la VaR, le CVaR devrait aussi être négatif

def test_get_statistics(sample_results : Results):
    # Générer les statistiques
    sample_results.get_statistics()
    # Vérifier si le DataFrame est bien généré
    assert isinstance(sample_results.df_statistics, pd.DataFrame)
    assert "Metrics" in sample_results.df_statistics.columns
    assert "Test Strategy" in sample_results.df_statistics.columns  # Le nom de la stratégie doit être dans les colonnes

def test_compare_results(sample_results : Results):
    # Créer un autre exemple de résultat pour la comparaison
    ptf_values_2 = pd.Series([90, 95, 100, 98, 105], index=pd.date_range('2023-01-01', periods=5, freq='D'))
    ptf_weights_2 = pd.DataFrame({
        'Asset1': [0.3, 0.44, 0.17, 0.6, 0.55],
        'Asset2': [0.7, 0.66, 0.83, 0.4, 0.45],
    }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
    results_2 = Results(ptf_values=ptf_values_2, 
                        strategy_name="Strategy 2",
                        ptf_weights= ptf_weights_2,
                        data_frequency=FrequencyType.DAILY)
    # Comparer les deux résultats
    combined_results = Results.compare_results([sample_results, results_2])
    
    # Vérifier si la comparaison a produit un résultat
    assert isinstance(combined_results, Results)
    assert "Metrics" in combined_results.df_statistics.columns

def test_create_plots(sample_results : Results):
    sample_results.create_plots()
    # Vérifier si les plots sont bien créés
    assert isinstance(sample_results.ptf_value_plot, go.Figure)
    assert isinstance(sample_results.ptf_drawdown_plot, go.Figure)
    assert isinstance(sample_results.ptf_weights_plot, go.Figure)
