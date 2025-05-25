import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    filename='performance_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    """Load portfolio returns and market data."""
    print("Loading data...")
    logging.info("Loading portfolio returns and market data")
    
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
    
    print(f"Portfolio date range: {portfolio_ret.index.min()} to {portfolio_ret.index.max()}")
    print(f"Market date range: {mkt_ind.index.min()} to {mkt_ind.index.max()}")
    print(f"Number of common dates: {len(common_dates)}")
    
    return portfolio_ret, mkt_ind

def calculate_return_metrics(portfolio_ret, mkt_ind):
    """Calculate return-based performance metrics."""
    print("\nCalculating return metrics...")
    logging.info("Calculating return metrics")
    
    # Monthly returns
    portfolio_monthly = portfolio_ret['portfolio_return']
    market_monthly = mkt_ind['sp_ret']
    rf_monthly = mkt_ind['rf']
    
    # Excess returns
    portfolio_excess = portfolio_monthly - rf_monthly
    market_excess = market_monthly - rf_monthly
    
    # Annualized metrics
    annual_factor = 12  # Monthly to annual
    
    # Calculate maximum drawdown properly
    # First calculate cumulative returns
    cum_returns = (1 + portfolio_monthly).cumprod()
    # Calculate running maximum
    running_max = cum_returns.expanding().max()
    # Calculate drawdowns
    drawdowns = (cum_returns / running_max) - 1
    # Get maximum drawdown
    max_drawdown = drawdowns.min()
    
    metrics = {
        'return_metrics': {
            'avg_monthly_return': float(portfolio_monthly.mean()),
            'return_volatility': float(portfolio_monthly.std()),
            'sharpe_ratio': float(portfolio_excess.mean() / portfolio_excess.std() * np.sqrt(annual_factor)),
            'max_drawdown': float(max_drawdown),
            'max_monthly_loss': float(portfolio_monthly.min())
        }
    }
    
    return metrics

def calculate_risk_metrics(portfolio_ret, mkt_ind):
    """Calculate risk-based performance metrics."""
    print("Calculating risk metrics...")
    logging.info("Calculating risk metrics")
    
    # Monthly returns
    portfolio_monthly = portfolio_ret['portfolio_return']
    market_monthly = mkt_ind['sp_ret']
    rf_monthly = mkt_ind['rf']
    
    # Excess returns
    portfolio_excess = portfolio_monthly - rf_monthly
    market_excess = market_monthly - rf_monthly
    
    # Calculate beta and alpha
    beta = np.cov(portfolio_excess, market_excess)[0,1] / np.var(market_excess)
    alpha = portfolio_excess.mean() - beta * market_excess.mean()
    
    # Information ratio
    tracking_error = (portfolio_excess - market_excess).std()
    information_ratio = (portfolio_excess.mean() - market_excess.mean()) / tracking_error
    
    metrics = {
        'risk_metrics': {
            'beta': float(beta),
            'alpha': float(alpha),
            'information_ratio': float(information_ratio)
        }
    }
    
    return metrics

def calculate_turnover(portfolio_ret):
    """Calculate true portfolio turnover based on permno changes."""
    print("Calculating turnover metrics...")
    logging.info("Calculating turnover metrics (true turnover)")
    
    # Parse permno lists
    long_lists = portfolio_ret['long_permnos'].apply(lambda x: set(x.split(',')))
    short_lists = portfolio_ret['short_permnos'].apply(lambda x: set(x.split(',')))
    
    turnovers = []
    for i in range(1, len(portfolio_ret)):
        prev_long = long_lists.iloc[i-1]
        prev_short = short_lists.iloc[i-1]
        curr_long = long_lists.iloc[i]
        curr_short = short_lists.iloc[i]
        # Fraction of positions changed
        long_turnover = 1 - len(prev_long & curr_long) / len(curr_long) if len(curr_long) > 0 else 0
        short_turnover = 1 - len(prev_short & curr_short) / len(curr_short) if len(curr_short) > 0 else 0
        total_turnover = (long_turnover + short_turnover) / 2
        turnovers.append(total_turnover)
    
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    metrics = {
        'turnover_metrics': {
            'avg_monthly_turnover': avg_turnover
        }
    }
    
    return metrics

def save_metrics(metrics):
    """Save performance metrics to JSON file."""
    print("\nSaving metrics to metrics.json...")
    logging.info("Saving performance metrics")
    
    try:
        with open('reports/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        # Ensure the file is closed and flushed if with statement doesn't guarantee it in this env
        # This is often implicit, but let's be explicit for debugging.
        # Python's `with` statement handles closing automatically.
        # If issues persist, this might indicate an environment-specific flushing issue.
    except Exception as e:
        print(f"Error saving metrics: {e}")
        logging.error(f"Error saving metrics: {e}", exc_info=True)
        raise
    
    print("Metrics saved successfully!")

def calculate_custom_oos_r2():
    """Calculate OOS R2 using the assignment-specific formula."""
    print("\nCalculating Custom OOS R2 (Assignment Formula)...")
    logging.info("Calculating Custom OOS R2 (Assignment Formula)")
    try:
        preds_df = pd.read_csv('ridge_oos_predictions.csv')
        # Ensure columns are numeric and handle potential errors
        preds_df['return'] = pd.to_numeric(preds_df['return'], errors='coerce')
        preds_df['prediction'] = pd.to_numeric(preds_df['prediction'], errors='coerce')
        preds_df.dropna(subset=['return', 'prediction'], inplace=True)

        if preds_df.empty:
            print("No valid data for OOS R2 calculation after cleaning.")
            logging.warning("No valid data for OOS R2 calculation after cleaning.")
            return {'custom_oos_r2': {'value': None, 'note': 'No valid data'}}

        sum_sq_errors = ((preds_df['return'] - preds_df['prediction'])**2).sum()
        sum_sq_actuals_centered_zero = (preds_df['return']**2).sum()

        if sum_sq_actuals_centered_zero == 0:
            print("Sum of squared actual returns (centered at zero) is zero. Cannot calculate OOS R2.")
            logging.warning("Sum of squared actual returns (centered at zero) is zero for OOS R2.")
            return {'custom_oos_r2': {'value': None, 'note': 'Denominator is zero'}}

        oos_r2 = 1 - (sum_sq_errors / sum_sq_actuals_centered_zero)
        print(f"Custom OOS R2: {oos_r2:.4f}")
        logging.info(f"Custom OOS R2: {oos_r2:.4f}")
        return {'custom_oos_r2': {'value': float(oos_r2)}}
    except FileNotFoundError:
        print("ridge_oos_predictions.csv not found. Cannot calculate OOS R2.")
        logging.error("ridge_oos_predictions.csv not found for OOS R2 calculation.")
        return {'custom_oos_r2': {'value': None, 'note': 'Prediction file not found'}}
    except Exception as e:
        print(f"Error calculating custom OOS R2: {e}")
        logging.error(f"Error calculating custom OOS R2: {e}", exc_info=True)
        return {'custom_oos_r2': {'value': None, 'note': f'Error: {e}'}}

def main():
    """Main function to evaluate portfolio performance."""
    try:
        # Load data
        portfolio_ret, mkt_ind = load_data()
        
        # Calculate metrics
        return_metrics = calculate_return_metrics(portfolio_ret, mkt_ind)
        risk_metrics = calculate_risk_metrics(portfolio_ret, mkt_ind)
        turnover_metrics = calculate_turnover(portfolio_ret)
        custom_r2_data = calculate_custom_oos_r2() # Calculate custom OOS R2
        
        # Combine all metrics
        all_metrics = {**return_metrics, **risk_metrics, **turnover_metrics}
        all_metrics['custom_oos_r2'] = custom_r2_data['custom_oos_r2'] # Explicitly add/overwrite
        
        # Save metrics
        save_metrics(all_metrics)
        
        # Print summary
        print("\nPerformance Summary:")
        print("\nReturn Metrics:")
        for metric, value in all_metrics['return_metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nRisk Metrics:")
        for metric, value in all_metrics['risk_metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nTurnover Metrics:")
        for metric, value in all_metrics['turnover_metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        if 'custom_oos_r2' in all_metrics and all_metrics['custom_oos_r2'].get('value') is not None:
            print("\nCustom OOS R2:")
            print(f"oos_r2_assignment_formula: {all_metrics['custom_oos_r2']['value']:.4f}")
        else:
            print("\nCustom OOS R2: Not calculated or error during calculation.")
            if 'custom_oos_r2' in all_metrics:
                 print(f"Note: {all_metrics['custom_oos_r2'].get('note')}")

    except Exception as e:
        logging.error(f"Error in performance evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 