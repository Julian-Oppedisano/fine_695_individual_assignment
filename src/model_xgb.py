import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

# Settings
WINDOWS = list(range(1, 15))  # Use all 14 windows
SPLIT_DIR = 'data/splits'
MODELS_DIR = 'models'
TARGET = 'stock_exret'
EXCLUDE_COLS = ['date', 'permno', 'stock_exret', 'shrcd', 'exchcd', 'stock_ticker', 'cusip', 'comp_name', 'year', 'month', 'rf']

# Reduced XGBoost parameters to tune
PARAM_GRID = {
    'n_estimators': [100, 500],  # Reduced from 3 to 2 values
    'max_depth': [3, 7],         # Reduced from 3 to 2 values
    'eta': [0.1, 0.3],          # Reduced from 3 to 2 values
    'subsample': [0.8],          # Fixed value
    'colsample_bytree': [0.8],   # Fixed value
    'min_child_weight': [3]      # Fixed value
}

def load_split(window, split):
    """Load data split for a specific window."""
    return pd.read_csv(os.path.join(SPLIT_DIR, f'{split}_window{window:02d}.csv'), index_col='date')

def get_X_y(df):
    """Prepare features and target for modeling."""
    features = [col for col in df.columns if col not in EXCLUDE_COLS]
    numeric_features = df[features].select_dtypes(include='number').columns.tolist()
    X = df[numeric_features].values
    y = df[TARGET].values
    return X, y, df.index

def tune_xgboost(X_train, y_train, X_val, y_val):
    """Tune XGBoost hyperparameters using validation set."""
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    # Calculate total combinations for progress bar
    total_combinations = (
        len(PARAM_GRID['n_estimators']) *
        len(PARAM_GRID['max_depth']) *
        len(PARAM_GRID['eta'])
    )
    
    # Create progress bar
    pbar = tqdm(total=total_combinations, desc='Tuning XGBoost')
    
    for n_estimators in PARAM_GRID['n_estimators']:
        for max_depth in PARAM_GRID['max_depth']:
            for eta in PARAM_GRID['eta']:
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'eta': eta,
                    'subsample': PARAM_GRID['subsample'][0],
                    'colsample_bytree': PARAM_GRID['colsample_bytree'][0],
                    'min_child_weight': PARAM_GRID['min_child_weight'][0],
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse'
                }
                
                # Train model with early stopping
                model = xgb.XGBRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                score = r2_score(y_val, val_pred)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                
                pbar.update(1)
    
    pbar.close()
    return best_model, best_params, best_score

def main():
    """Main function to train and evaluate XGBoost models."""
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Store results
    results = []
    all_predictions = []
    
    # Create progress bar for windows
    window_pbar = tqdm(WINDOWS, desc='Processing Windows')
    
    for window in window_pbar:
        window_pbar.set_description(f'Processing Window {window}')
        
        # Load data
        train = load_split(window, 'train')
        val = load_split(window, 'val')
        test = load_split(window, 'test')
        
        # Prepare features and target
        X_train, y_train, train_idx = get_X_y(train)
        X_val, y_val, val_idx = get_X_y(val)
        X_test, y_test, test_idx = get_X_y(test)
        
        # Tune and train model
        best_model, best_params, val_score = tune_xgboost(X_train, y_train, X_val, y_val)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'xgb_window{window:02d}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save best parameters
        params_path = os.path.join(MODELS_DIR, f'xgb_params_window{window:02d}.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f)
        
        # Generate predictions
        val_pred = best_model.predict(X_val)
        test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Store predictions
        val_pred_df = pd.DataFrame({
            'date': val_idx,
            'permno': val['permno'],
            'prediction': val_pred,
            'return': val[TARGET]  # Include actual returns
        })
        test_pred_df = pd.DataFrame({
            'date': test_idx,
            'permno': test['permno'],
            'prediction': test_pred,
            'return': test[TARGET]  # Include actual returns
        })
        all_predictions.extend([val_pred_df, test_pred_df])
        
        # Store results
        results.append({
            'window': window,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'best_params': best_params
        })
        
        # Update progress bar description with current metrics
        window_pbar.set_postfix({
            'Val R²': f'{val_r2:.4f}',
            'Test R²': f'{test_r2:.4f}'
        })
    
    window_pbar.close()
    
    # Save all predictions
    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_df.to_csv('preds_xgb.csv', index=False)
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('xgb_results.csv', index=False)
    
    # Print final summary
    print('\n--- Final Summary ---')
    print(f'Average Validation R²: {results_df["val_r2"].mean():.4f}')
    print(f'Average Test R²: {results_df["test_r2"].mean():.4f}')

if __name__ == '__main__':
    main() 