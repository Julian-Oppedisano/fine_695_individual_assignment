import pandas as pd
import numpy as np
import os
import time
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import csv

# Settings
WINDOWS = list(range(1, 15))  # Use all 14 windows
SPLIT_DIR = 'data/splits'
TARGET = 'stock_exret'
EXCLUDE_COLS = ['date', 'permno', 'stock_exret', 'shrcd', 'exchcd', 'stock_ticker', 'cusip', 'comp_name', 'year', 'month', 'rf']

# Risk management parameters
MAX_POSITION_SIZE = 0.05  # Maximum 5% position size per stock
STOP_LOSS = 0.02  # 2% stop loss per position
MAX_PORTFOLIO_VOL = 0.15  # Maximum 15% annualized portfolio volatility
CONFIDENCE_THRESHOLD = 0.01  # Minimum prediction magnitude to take a position
MAX_SECTOR_EXPOSURE = 0.25  # Maximum 25% exposure to any sector
MIN_CORRELATION_THRESHOLD = 0.3  # Minimum correlation threshold for position sizing
LOOKBACK_PERIOD = 60  # Days for volatility and correlation calculations
MARKET_REGIME_THRESHOLD = 0.1  # Threshold for market regime detection

# Small grid for speed
PARAM_GRIDS = {
    'lasso': {'alpha': [0.01, 0.1, 1]},
    'ridge': {'alpha': [0.01, 0.1, 1]},
    'enet':  {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}
}

MODELS = {
    'lasso': Lasso(max_iter=1000),
    'ridge': Ridge(max_iter=1000),
    'enet': ElasticNet(max_iter=1000)
}

results = []
all_ridge_predictions = [] # Initialize list to store all Ridge predictions

def load_split(window, split):
    return pd.read_csv(os.path.join(SPLIT_DIR, f'{split}_window{window:02d}.csv'), index_col='date')

def get_X_y(df):
    features = [col for col in df.columns if col not in EXCLUDE_COLS]
    # Only use numeric columns
    numeric_features = df[features].select_dtypes(include='number').columns.tolist()
    X = df[numeric_features].values
    y = df[TARGET].values
    return X, y, df.index, df['permno'].values

def detect_market_regime(returns):
    """Detect market regime based on returns."""
    # Calculate rolling metrics
    vol = returns.rolling(window=LOOKBACK_PERIOD).std() * np.sqrt(12)
    trend = returns.rolling(window=LOOKBACK_PERIOD).mean()
    
    # Initialize regime array
    regime = pd.Series(index=returns.index, dtype='object')
    
    # Define regimes
    regime[vol > vol.mean() + vol.std()] = 'high_vol'
    regime[(vol <= vol.mean() + vol.std()) & (trend > MARKET_REGIME_THRESHOLD)] = 'uptrend'
    regime[(vol <= vol.mean() + vol.std()) & (trend < -MARKET_REGIME_THRESHOLD)] = 'downtrend'
    regime[regime.isna()] = 'normal'
    
    return regime

def calculate_position_sizes(predictions, returns, confidence, correlation_weights, sector_exposure=None):
    """Calculate position sizes with enhanced risk management."""
    # Initialize position sizes
    position_sizes = pd.Series(0.0, index=predictions.index)
    # Calculate base position sizes using prediction direction and confidence
    position_sizes = np.sign(predictions) * confidence
    # Apply correlation weights
    position_sizes *= correlation_weights
    # Apply position limits
    position_sizes = position_sizes.clip(-MAX_POSITION_SIZE, MAX_POSITION_SIZE)
    # Apply stop-loss
    stop_loss_mask = returns < -STOP_LOSS
    position_sizes[stop_loss_mask] = 0
    # Normalize to sum to 1
    if position_sizes.abs().sum() > 0:
        position_sizes = position_sizes / position_sizes.abs().sum()
    return position_sizes

def analyze_portfolio_performance(predictions, returns, model_name, window, index):
    """Analyze portfolio performance with enhanced metrics."""
    # Convert numpy arrays to pandas Series with correct index
    predictions = pd.Series(predictions, index=index)
    returns = pd.Series(returns, index=index)
    # Calculate confidence based on prediction magnitude
    confidence = predictions.abs() / predictions.abs().max()
    # Calculate correlation weights
    correlation_weights = pd.Series(1.0, index=predictions.index)
    for date in returns.index.unique():
        date_mask = returns.index == date
        if date_mask.sum() > 0:
            corr = np.corrcoef(predictions[date_mask], returns[date_mask])[0, 1]
            correlation_weights[date_mask] = max(0, corr) if not np.isnan(corr) else 0
    # Calculate position sizes
    position_sizes = calculate_position_sizes(
        predictions, returns, confidence, correlation_weights
    )
    # Calculate portfolio returns
    portfolio_returns = (position_sizes * returns).groupby('date').sum()
    # Calculate performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
    win_rate = (portfolio_returns > 0).mean()
    # Calculate drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    # Calculate turnover
    position_changes = position_sizes.groupby('date').diff().abs()
    turnover = position_changes.groupby('date').sum().mean()
    # Calculate regime-specific metrics
    regime_returns = {}
    regime_sharpe = {}
    # Print results
    print(f"\nModel: {model_name}")
    print(f"Val Portfolio - Return: {total_return:.4f}, Sharpe: {sharpe_ratio:.4f}, Win Rate: {win_rate:.4f}")
    print(f"Val Portfolio - Max DD: {max_drawdown:.4f}, Turnover: {turnover:.4f}, Vol: {volatility:.4f}")
    print(f"Val Portfolio - Regime Returns: {regime_returns}")
    print(f"Val Portfolio - Regime Sharpe: {regime_sharpe}")
    return {
        'val_return': total_return,
        'val_sharpe': sharpe_ratio,
        'val_win_rate': win_rate,
        'val_max_dd': max_drawdown,
        'val_turnover': turnover,
        'val_vol': volatility,
        'val_regime_returns': regime_returns,
        'val_regime_sharpe': regime_sharpe
    }

for window in WINDOWS:
    print(f'\n--- Window {window} ---')
    train = load_split(window, 'train')
    val = load_split(window, 'val')
    test = load_split(window, 'test')
    X_train, y_train, train_idx, permnos_train = get_X_y(train)
    X_val, y_val, val_idx, permnos_val = get_X_y(val)
    X_test, y_test, test_idx, permnos_test = get_X_y(test)
    
    for model_name, model in MODELS.items():
        print(f'\nModel: {model_name.upper()}')
        start = time.time()
        grid = GridSearchCV(model, PARAM_GRIDS[model_name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f'Best params: {grid.best_params_}')
        
        # Get predictions
        val_pred = best_model.predict(X_val)
        test_pred = best_model.predict(X_test)
        
        if model_name == 'ridge':
            # Store Ridge predictions for the current window's test set
            current_ridge_preds = pd.DataFrame({
                'date': test_idx,
                'permno': permnos_test,
                'prediction': test_pred,
                'return': y_test
            })
            all_ridge_predictions.append(current_ridge_preds)
        
        # Calculate traditional metrics
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Calculate portfolio performance with risk management
        val_portfolio = analyze_portfolio_performance(val_pred, y_val, model_name, window, val_idx)
        test_portfolio = analyze_portfolio_performance(test_pred, y_test, model_name, window, test_idx)
        
        elapsed = time.time() - start
        
        print(f'Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}')
        print(f'Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}')
        print(f'Val Portfolio - Return: {val_portfolio["val_return"]:.4f}, Sharpe: {val_portfolio["val_sharpe"]:.4f}, Win Rate: {val_portfolio["val_win_rate"]:.4f}')
        print(f'Val Portfolio - Max DD: {val_portfolio["val_max_dd"]:.4f}, Turnover: {val_portfolio["val_turnover"]:.4f}, Vol: {val_portfolio["val_vol"]:.4f}')
        print('Val Portfolio - Regime Returns:', {k: f'{v:.4f}' for k, v in val_portfolio['val_regime_returns'].items()})
        print('Val Portfolio - Regime Sharpe:', {k: f'{v:.4f}' for k, v in val_portfolio['val_regime_sharpe'].items()})
        print(f'Test Portfolio - Return: {test_portfolio["val_return"]:.4f}, Sharpe: {test_portfolio["val_sharpe"]:.4f}, Win Rate: {test_portfolio["val_win_rate"]:.4f}')
        print(f'Test Portfolio - Max DD: {test_portfolio["val_max_dd"]:.4f}, Turnover: {test_portfolio["val_turnover"]:.4f}, Vol: {test_portfolio["val_vol"]:.4f}')
        print('Test Portfolio - Regime Returns:', {k: f'{v:.4f}' for k, v in test_portfolio['val_regime_returns'].items()})
        print('Test Portfolio - Regime Sharpe:', {k: f'{v:.4f}' for k, v in test_portfolio['val_regime_sharpe'].items()})
        print(f'Elapsed: {elapsed:.1f} sec')
        
        results.append({
            'window': window,
            'model': model_name,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'val_return': val_portfolio['val_return'],
            'test_return': test_portfolio['val_return'],
            'val_sharpe': val_portfolio['val_sharpe'],
            'test_sharpe': test_portfolio['val_sharpe'],
            'val_win_rate': val_portfolio['val_win_rate'],
            'test_win_rate': test_portfolio['val_win_rate'],
            'val_max_dd': val_portfolio['val_max_dd'],
            'test_max_dd': test_portfolio['val_max_dd'],
            'val_turnover': val_portfolio['val_turnover'],
            'test_turnover': test_portfolio['val_turnover'],
            'val_vol': val_portfolio['val_vol'],
            'test_vol': test_portfolio['val_vol'],
            'val_regime_returns': val_portfolio['val_regime_returns'],
            'test_regime_returns': test_portfolio['val_regime_returns'],
            'val_regime_sharpe': val_portfolio['val_regime_sharpe'],
            'test_regime_sharpe': test_portfolio['val_regime_sharpe'],
            'best_params': grid.best_params_,
            'elapsed_sec': elapsed
        })

# Print summary table
print('\n--- Summary ---')
summary = pd.DataFrame(results)
print(summary[['window', 'model', 'val_return', 'test_return', 'val_sharpe', 'test_sharpe', 
               'val_win_rate', 'test_win_rate', 'val_max_dd', 'test_max_dd', 
               'val_turnover', 'test_turnover', 'val_vol', 'test_vol']])

# After the main for loop, save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('baseline_results.csv', index=False)
print('Saved baseline OOS R² results to baseline_results.csv')

# Concatenate and save all Ridge predictions after all windows are processed
if all_ridge_predictions:
    final_ridge_predictions_df = pd.concat(all_ridge_predictions, ignore_index=True)
    # Ensure date column is in YYYY-MM-DD format if it's not already
    final_ridge_predictions_df['date'] = pd.to_datetime(final_ridge_predictions_df['date']).dt.strftime('%Y-%m-%d')
    output_filename = 'ridge_oos_predictions.csv'
    final_ridge_predictions_df.to_csv(output_filename, index=False)
    print(f'Saved all Ridge OOS predictions to {output_filename}')
else:
    print('No Ridge predictions were collected.') 