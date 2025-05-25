import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import json
from datetime import datetime
import yfinance as yf

def load_data():
    # Load portfolio returns
    portfolio_ret = pd.read_csv('portfolio_ret.csv')
    portfolio_ret['date'] = pd.to_datetime(portfolio_ret['date'])
    
    # Load market data
    market_data = pd.read_csv('data/raw/mkt_ind.csv')
    # Create a date column from year and month
    market_data['date'] = pd.to_datetime(dict(year=market_data['year'], month=market_data['month'], day=1))
    
    # Load metrics
    with open('reports/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    return portfolio_ret, market_data, metrics

def create_macro_analysis(portfolio_ret, market_data):
    # Get S&P 500 data for the period
    sp500 = yf.download('^GSPC', start=portfolio_ret['date'].min(), end=portfolio_ret['date'].max(), interval='1mo')
    
    # Identify major market events
    events = {
        '2010-05-06': 'Flash Crash',
        '2011-08-05': 'US Credit Rating Downgrade',
        '2015-08-24': 'Chinese Market Crash',
        '2018-12-24': 'Christmas Eve Crash',
        '2020-03-23': 'COVID-19 Market Bottom',
        '2022-01-03': 'Inflation Concerns Begin'
    }
    
    # Create cumulative returns plot
    plt.figure(figsize=(12, 6))
    portfolio_cumret = (1 + portfolio_ret['portfolio_return']).cumprod()
    # Use the correct MultiIndex column for Close prices
    sp500_close = sp500[('Close', '^GSPC')]
    sp500_cumret = (1 + sp500_close.pct_change()).cumprod()
    
    plt.plot(portfolio_ret['date'], portfolio_cumret, label='Portfolio')
    plt.plot(sp500.index, sp500_cumret, label='S&P 500')
    
    # Add event markers
    for date, event in events.items():
        plt.axvline(pd.to_datetime(date), color='gray', linestyle='--', alpha=0.5)
        plt.text(pd.to_datetime(date), plt.ylim()[0], event, rotation=45, ha='right')
    
    plt.title('Portfolio Performance vs S&P 500 with Major Market Events')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/macro_analysis.png')
    plt.close()

def analyze_top_holdings(portfolio_ret):
    # Calculate average returns and volatility for top holdings
    holdings_dict = {}
    
    for col in ['long_permnos', 'short_permnos']:
        if col in portfolio_ret.columns:
            # Convert string representation of lists to actual lists
            portfolio_ret[col] = portfolio_ret[col].apply(eval)
            
            # Calculate statistics for each holding
            for _, row in portfolio_ret.iterrows():
                for permno in row[col]:
                    if permno not in holdings_dict:
                        holdings_dict[permno] = []
                    holdings_dict[permno].append(row['portfolio_return'])
    
    # Convert to DataFrame
    holdings_analysis = pd.DataFrame({
        'permno': list(holdings_dict.keys()),
        'returns': list(holdings_dict.values())
    })
    holdings_analysis['avg_return'] = holdings_analysis['returns'].apply(np.mean)
    holdings_analysis['volatility'] = holdings_analysis['returns'].apply(np.std)
    holdings_analysis['sharpe'] = holdings_analysis['avg_return'] / holdings_analysis['volatility']
    
    # Sort by average return and get top 10
    top_10 = holdings_analysis.nlargest(10, 'avg_return')
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_10)), top_10['avg_return'])
    plt.xticks(range(len(top_10)), top_10['permno'], rotation=45)
    plt.title('Top 10 Holdings by Average Return')
    plt.xlabel('Permno')
    plt.ylabel('Average Return')
    plt.tight_layout()
    plt.savefig('reports/top_holdings_analysis.png')
    plt.close()
    
    return top_10

def create_performance_visualizations(portfolio_ret, metrics):
    # Create rolling performance plot
    plt.figure(figsize=(12, 6))
    rolling_ret = portfolio_ret['portfolio_return'].rolling(window=12).mean()
    plt.plot(portfolio_ret['date'], rolling_ret)
    plt.title('12-Month Rolling Average Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/rolling_performance.png')
    plt.close()
    
    # Create drawdown plot
    cumulative_returns = (1 + portfolio_ret['portfolio_return']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_ret['date'], drawdown)
    plt.title('Portfolio Drawdown')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/drawdown_analysis.png')
    plt.close()

def main():
    # Load data
    portfolio_ret, market_data, metrics = load_data()
    
    # Create enhanced visualizations
    create_macro_analysis(portfolio_ret, market_data)
    top_holdings = analyze_top_holdings(portfolio_ret)
    create_performance_visualizations(portfolio_ret, metrics)
    
    print("Enhanced visualizations and analysis generated successfully!")

if __name__ == "__main__":
    main() 