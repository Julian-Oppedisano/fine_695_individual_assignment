import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Settings
PREDICTIONS_FILE = 'ridge_oos_predictions.csv'
OUTPUT_FILE = 'portfolio_ret.csv'
LOG_FILE = 'portfolio_construction.log'
TOP_N = 50  # Number of stocks to long/short
EQUAL_WEIGHT = 1.0 / TOP_N  # Equal weight for each position
MIN_STOCKS = 100  # Minimum number of stocks required for portfolio construction

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_predictions():
    """Load Ridge model predictions."""
    print("Loading predictions...")
    logging.info("Loading predictions from %s", PREDICTIONS_FILE)
    preds = pd.read_csv(PREDICTIONS_FILE)
    preds['date'] = pd.to_datetime(preds['date'])
    return preds

def construct_portfolio(preds):
    """Construct long-short portfolio based on predictions."""
    print("Constructing portfolio...")
    logging.info("Starting portfolio construction")
    
    # Initialize list to store portfolio returns
    portfolio_returns = []
    skipped_months = []
    
    # Get all unique dates in the predictions
    all_dates = pd.date_range(preds['date'].min(), preds['date'].max(), freq='M')
    
    # Group by date and process each month
    for date in all_dates:
        # Get predictions for this month
        month_preds = preds[preds['date'].dt.to_period('M') == date.to_period('M')]
        
        # Log the number of stocks available
        n_stocks = len(month_preds)
        logging.info(f"Date: {date.strftime('%Y-%m-%d')}, Stocks available: {n_stocks}")
        
        # Skip if not enough stocks
        if n_stocks < MIN_STOCKS:
            skipped_months.append({
                'date': date,
                'n_stocks': n_stocks,
                'reason': 'Insufficient stocks'
            })
            logging.warning(f"Skipping {date.strftime('%Y-%m-%d')} - Only {n_stocks} stocks available")
            continue
        
        # Sort by predictions
        sorted_preds = month_preds.sort_values('prediction', ascending=False)
        
        # Get top and bottom N stocks
        top_stocks = sorted_preds.head(TOP_N)
        bottom_stocks = sorted_preds.tail(TOP_N)
        
        # Calculate portfolio return using actual returns
        # Long top N stocks, short bottom N stocks
        portfolio_return = (
            top_stocks['return'].mean() -  # Long position
            bottom_stocks['return'].mean()  # Short position
        )
        
        # Store results
        portfolio_returns.append({
            'date': date,
            'portfolio_return': portfolio_return,
            'long_count': len(top_stocks),
            'short_count': len(bottom_stocks),
            'long_mean_pred': top_stocks['prediction'].mean(),
            'short_mean_pred': bottom_stocks['prediction'].mean(),
            'total_stocks': n_stocks,
            'long_permnos': ','.join(str(x) for x in top_stocks['permno']),
            'short_permnos': ','.join(str(x) for x in bottom_stocks['permno'])
        })
        
        logging.info(f"Portfolio constructed for {date.strftime('%Y-%m-%d')} - Return: {portfolio_return:.4f}")
    
    # Convert to DataFrame
    portfolio_df = pd.DataFrame(portfolio_returns)
    if not portfolio_df.empty:
        portfolio_df.set_index('date', inplace=True)
    
    # Log summary of skipped months
    if skipped_months:
        skipped_df = pd.DataFrame(skipped_months)
        logging.info("\nSkipped Months Summary:")
        logging.info(f"Total months skipped: {len(skipped_months)}")
        logging.info("\nSkipped months details:")
        for month in skipped_months:
            logging.info(f"Date: {month['date'].strftime('%Y-%m-%d')}, Stocks: {month['n_stocks']}, Reason: {month['reason']}")
    
    return portfolio_df

def save_portfolio_returns(portfolio_df):
    """Save portfolio returns to CSV."""
    print(f"Saving portfolio returns to {OUTPUT_FILE}...")
    logging.info(f"Saving portfolio returns to {OUTPUT_FILE}")
    
    if portfolio_df.empty:
        logging.error("No portfolio returns to save!")
        return
    
    portfolio_df.to_csv(OUTPUT_FILE)
    
    # Print and log summary statistics
    summary = {
        'Date Range': f"{portfolio_df.index.min()} to {portfolio_df.index.max()}",
        'Number of Months': len(portfolio_df),
        'Average Monthly Return': portfolio_df['portfolio_return'].mean(),
        'Return Volatility': portfolio_df['portfolio_return'].std(),
        'Sharpe Ratio': (portfolio_df['portfolio_return'].mean() / portfolio_df['portfolio_return'].std()),
        'Average Stocks per Month': portfolio_df['total_stocks'].mean()
    }
    
    print("\nPortfolio Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
        logging.info(f"{key}: {value}")

def main():
    """Main function to construct and save portfolio returns."""
    try:
        # Load predictions
        preds = load_predictions()
        
        # Construct portfolio
        portfolio_df = construct_portfolio(preds)
        
        # Save results
        save_portfolio_returns(portfolio_df)
        
    except Exception as e:
        logging.error(f"Error in portfolio construction: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 