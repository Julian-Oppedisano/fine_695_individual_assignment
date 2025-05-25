import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load portfolio returns and market data."""
    print("Loading data...")
    
    # Load portfolio returns
    portfolio_ret = pd.read_csv('portfolio_ret.csv')
    portfolio_ret['date'] = pd.to_datetime(portfolio_ret['date'])
    portfolio_ret.set_index('date', inplace=True)
    
    # Load market data
    mkt_ind = pd.read_csv('data/raw/mkt_ind.csv')
    mkt_ind['date'] = pd.to_datetime(mkt_ind[['year', 'month']].assign(day=1)) + pd.offsets.MonthEnd(0)
    mkt_ind.set_index('date', inplace=True)
    
    # Align dates
    common_dates = portfolio_ret.index.intersection(mkt_ind.index)
    portfolio_ret = portfolio_ret.loc[common_dates]
    mkt_ind = mkt_ind.loc[common_dates]
    
    return portfolio_ret, mkt_ind

def plot_cumulative_returns(portfolio_ret, mkt_ind):
    """Plot cumulative returns of portfolio vs market."""
    print("Plotting cumulative returns...")
    
    # Calculate cumulative returns
    portfolio_cumret = (1 + portfolio_ret['portfolio_return']).cumprod()
    market_cumret = (1 + mkt_ind['sp_ret']).cumprod()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumret.index, portfolio_cumret, label='Portfolio', linewidth=2)
    plt.plot(market_cumret.index, market_cumret, label='S&P 500', linewidth=2, alpha=0.7)
    plt.title('Cumulative Returns: Portfolio vs S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/macro_analysis.png')
    plt.close()

def plot_rolling_sharpe(portfolio_ret, mkt_ind, window=12):
    """Plot rolling Sharpe ratio."""
    print("Plotting rolling Sharpe ratio...")
    
    # Calculate excess returns
    portfolio_excess = portfolio_ret['portfolio_return'] - mkt_ind['rf']
    
    # Calculate rolling Sharpe ratio
    rolling_sharpe = (
        portfolio_excess.rolling(window=window).mean() /
        portfolio_excess.rolling(window=window).std() *
        np.sqrt(12)  # Annualize
    )
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
    plt.title(f'Rolling {window}-Month Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/rolling_sharpe.png')
    plt.close()

def plot_drawdown(portfolio_ret):
    """Plot portfolio drawdown over time."""
    print("Plotting drawdown...")
    
    # Calculate drawdown
    cum_returns = (1 + portfolio_ret['portfolio_return']).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns / running_max) - 1
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='red', linewidth=2)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/drawdown_analysis.png')
    plt.close()

def plot_returns_distribution(portfolio_ret, mkt_ind):
    """Plot distribution of monthly returns."""
    print("Plotting returns distribution...")
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.histplot(portfolio_ret['portfolio_return'], label='Portfolio', alpha=0.5, bins=50)
    sns.histplot(mkt_ind['sp_ret'], label='S&P 500', alpha=0.5, bins=50)
    plt.title('Distribution of Monthly Returns')
    plt.xlabel('Monthly Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/returns_distribution.png')
    plt.close()

def plot_rolling_correlation(portfolio_ret, mkt_ind, window=12):
    """Plot rolling correlation with market."""
    print("Plotting rolling correlation...")
    
    # Calculate rolling correlation
    rolling_corr = portfolio_ret['portfolio_return'].rolling(window=window).corr(mkt_ind['sp_ret'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr.index, rolling_corr, linewidth=2)
    plt.title(f'Rolling {window}-Month Correlation with S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/rolling_correlation.png')
    plt.close()

def plot_decile_spread():
    """Plot decile-spread (P10–P1) line chart."""
    print("Plotting decile-spread (P10–P1) line chart...")
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Load predictions with actual returns
    preds = pd.read_csv('ridge_oos_predictions.csv')
    preds['date'] = pd.to_datetime(preds['date'])
    
    # For each month, assign deciles by prediction
    def assign_deciles(df):
        df = df.copy()
        df['decile'] = pd.qcut(df['prediction'], 10, labels=False) + 1  # 1 to 10
        return df
    
    preds = preds.groupby('date', group_keys=False).apply(assign_deciles)
    
    # Calculate mean actual return for each decile per month
    decile_returns = preds.groupby(['date', 'decile'])['return'].mean().unstack()
    
    # Compute decile spread (P10 - P1)
    decile_spread = decile_returns[10] - decile_returns[1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(decile_spread.index, decile_spread, label='P10 - P1 Spread', color='purple')
    plt.title('Monthly Decile-Spread (P10–P1) Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (P10 - P1)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/decile_spread.png')
    plt.close()

def plot_top_holdings_by_avg_actual_return(portfolio_ret_path='portfolio_ret.csv', predictions_path='ridge_oos_predictions.csv'):
    """Plot top 10 holdings by average actual return when held in the long portfolio."""
    print("Plotting Top 10 Holdings by Average Actual Return (when held long)...")
    logging.info("Plotting Top 10 Holdings by Average Actual Return (when held long)")

    try:
        portfolio_df = pd.read_csv(portfolio_ret_path)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        predictions_df = pd.read_csv(predictions_path)
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])

        held_long_returns = []

        for _, row in portfolio_df.iterrows():
            current_date = row['date']
            if pd.isna(row['long_permnos']): # Handle cases with no long positions for a month
                continue
            long_permnos_this_month = set(str(row['long_permnos']).split(','))
            
            # Filter predictions for the current date and permnos held long
            actuals_for_month = predictions_df[
                (predictions_df['date'] == current_date) &
                (predictions_df['permno'].astype(str).isin(long_permnos_this_month))
            ]
            
            for _, actual_row in actuals_for_month.iterrows():
                held_long_returns.append({
                    'permno': actual_row['permno'],
                    'return': actual_row['return']
                })

        if not held_long_returns:
            print("No long holdings found to analyze for top holdings plot.")
            logging.warning("No long holdings found for top holdings plot.")
            # Create an empty plot or a plot with a message
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No long holdings data available to plot Top 10 Holdings.', horizontalalignment='center', verticalalignment='center')
            plt.title('Top 10 Holdings by Average Actual Return (when held long)')
            plt.savefig('reports/top_holdings_analysis.png')
            plt.close()
            return

        held_long_returns_df = pd.DataFrame(held_long_returns)
        avg_returns_when_held = held_long_returns_df.groupby('permno')['return'].mean().sort_values(ascending=False)
        
        top_10_holdings = avg_returns_when_held.head(10)

        plt.figure(figsize=(12, 7))
        top_10_holdings.plot(kind='bar')
        plt.title('Top 10 Holdings by Average Actual Return (when held long in portfolio)')
        plt.xlabel('Permno')
        plt.ylabel('Average Actual Monthly Return (when held)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('reports/top_holdings_analysis.png')
        plt.close()
        logging.info("Successfully plotted top 10 holdings.")

    except Exception as e:
        print(f"Error plotting top holdings: {str(e)}")
        logging.error(f"Error plotting top holdings: {str(e)}", exc_info=True)
        # Create an empty plot or a plot with a message in case of error
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error generating Top 10 Holdings plot: {str(e)}', horizontalalignment='center', verticalalignment='center', wrap=True)
        plt.title('Top 10 Holdings by Average Actual Return (when held long)')
        plt.savefig('reports/top_holdings_analysis.png')
        plt.close()

def main():
    """Main function to generate all visualizations."""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    try:
        # Load data
        portfolio_ret, mkt_ind = load_data()
        
        # Generate plots
        plot_cumulative_returns(portfolio_ret, mkt_ind)
        plot_rolling_sharpe(portfolio_ret, mkt_ind)
        plot_drawdown(portfolio_ret)
        plot_returns_distribution(portfolio_ret, mkt_ind)
        plot_rolling_correlation(portfolio_ret, mkt_ind)
        plot_decile_spread()
        plot_top_holdings_by_avg_actual_return()
        
        print("\nAll plots have been generated in the 'plots' and 'figures' directories.")
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        raise

if __name__ == '__main__':
    main() 