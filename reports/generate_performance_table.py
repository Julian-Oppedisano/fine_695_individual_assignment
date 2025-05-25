import pandas as pd
import numpy as np
import json

# Load your portfolio metrics
with open('reports/metrics.json', 'r') as f:
    metrics = json.load(f)

# Calculate S&P 500 metrics
sp500 = pd.read_csv('data/raw/mkt_ind.csv')
sp500['date'] = pd.to_datetime(sp500[['year', 'month']].assign(day=1))
sp500 = sp500[(sp500['date'] >= '2010-01-01') & (sp500['date'] <= '2023-12-31')]
sp500_avg_ann = sp500['sp_ret'].mean() * 12
sp500_std_ann = sp500['sp_ret'].std() * np.sqrt(12)
sp500_sharpe = sp500_avg_ann / sp500_std_ann

# Build table
metrics_table = {
    'Metric': [
        'Annualized Return', 'Annualized Std Dev', 'Sharpe Ratio',
        'Alpha (annualized)', 'Information Ratio', 'Max Drawdown', 'Max 1-mo Loss', 'Turnover'
    ],
    'Strategy': [
        metrics['return_metrics']['avg_monthly_return'] * 12,
        metrics['return_metrics']['return_volatility'] * np.sqrt(12),
        metrics['return_metrics']['sharpe_ratio'],
        metrics['risk_metrics']['alpha'] * 12,
        metrics['risk_metrics']['information_ratio'],
        metrics['return_metrics']['max_drawdown'],
        metrics['return_metrics']['max_monthly_loss'],
        metrics['turnover_metrics']['avg_monthly_turnover']
    ],
    'S&P 500': [
        sp500_avg_ann, sp500_std_ann, sp500_sharpe,
        '', '', '', '', ''
    ]
}
df = pd.DataFrame(metrics_table)
print(df.to_markdown(index=False))
df.to_csv('reports/performance_table.csv', index=False) 