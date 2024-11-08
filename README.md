# Risk Analytics Project

## Overview

The Risk Analytics project is designed to provide a comprehensive simulation of financial transaction data, with a focus on identifying high-risk transactions. This project generates synthetic data, applies machine learning models to classify transaction risk, and creates visualizations to analyze and interpret risk patterns.

## Goal

The primary goal of this project is to create a risk assessment model that can help financial institutions identify and analyze potential high-risk transactions based on various indicators such as compliance score, payment delay, and transaction volatility.

## Dataset

The dataset used in this project is synthetically generated to simulate real-world financial transactions. Key features include:

- **Transaction Features**: Amount, frequency, tenure, credit score, payment delays.
- **Risk Indicators**: Labels are generated based on volatility, delays, and compliance thresholds.

This data structure allows for robust analysis and classification of transactions by risk level, making it ideal for training and testing risk assessment models.

## Visualizations

The project generates several visualizations saved as HTML files in the `visualizations` directory, providing insights into various aspects of risk:

1. **Risk Distribution**: Proportion of low- versus high-risk transactions.
2. **Industry Analysis**: Average transaction amounts and risk levels across different industry sectors.
3. **Compliance Dashboard**: Analysis of compliance scores and their relationship with risk.
4. **Temporal Analysis**: Patterns of risk over time, including daily and monthly fluctuations.

## Model Training

The project includes a machine learning model, trained using features from the generated data to classify transactions based on their risk levels. The model and scaler are saved in the `models` directory for easy access and further analysis.

## Usage

1. Clone the repository.
2. Set up the virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
