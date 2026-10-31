# SupplyGuard AI

### Predictive Supply Chain Risk & Decision Intelligence Platform

SupplyGuard AI is an AI-powered supply chain analytics platform designed to help manufacturers anticipate demand, identify inventory shortages, assess supplier risks, and make proactive replenishment decisions.

## Problem Statement

Manufacturers face demand fluctuations, inventory shortages, supplier delays, and limited visibility across their supply chains. Traditional reactive planning can result in production disruptions, excess inventory, and increased operational costs.

SupplyGuard AI addresses these challenges through predictive demand forecasting and explainable risk analysis.

## Key Features

- **12-Week Demand Forecasting:** XGBoost-based demand prediction.
- **Inventory Risk Detection:** Identifies components at risk of stock-outs.
- **Supplier Risk Intelligence:** Evaluates supplier reliability and lead times.
- **Decision Intelligence:** Generates actionable inventory and supplier recommendations.
- **Interactive Dashboard:** Visualizes forecasts, inventory risks, and supplier performance.

## Technology Stack

- Python 3.11
- Exasol Personal
- PyExasol
- XGBoost
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly

## Architecture

1. Synthetic supply chain data is generated for demonstration.
2. Demand history is stored and retrieved through Exasol.
3. Python processes historical demand and supplier information.
4. XGBoost generates 12-week demand forecasts.
5. Inventory and supplier risk scores are calculated.
6. Streamlit displays insights and recommended actions.

## Project Structure

```text
SupplyGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── data/
├── docs/
├── pitch/
├── sql/
│   └── schema.sql
│
└── src/
    ├── __init__.py
    ├── data_generator.py
    ├── exasol_manager.py
    └── analysis_engine.py