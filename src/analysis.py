import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob
import pickle

def identify_top_holdings():
    """Identify average top 10 holdings across OOS and save as markdown table."""
    print("Identifying average top 10 holdings across OOS...")
    # Load predictions with actual returns
    preds = pd.read_csv('preds_xgb.csv')
    preds['date'] = pd.to_datetime(preds['date'])
    
    # For each month, assign deciles by prediction
    def assign_deciles(df):
        df = df.copy()
        df['decile'] = pd.qcut(df['prediction'], 10, labels=False) + 1  # 1 to 10
        return df
    
    preds = preds.groupby('date', group_keys=False).apply(assign_deciles)
    
    # Filter for top decile (P10)
    top_decile = preds[preds['decile'] == 10]
    
    # Group by permno and calculate average actual return
    avg_returns = top_decile.groupby('permno')['return'].mean().reset_index()
    avg_returns = avg_returns.sort_values('return', ascending=False)
    
    # Select top 10 holdings
    top_10 = avg_returns.head(10)
    
    # Save as markdown table
    os.makedirs('reports', exist_ok=True)
    with open('reports/top_10_holdings.md', 'w') as f:
        f.write('# Average Top 10 Holdings Across OOS\n\n')
        f.write('| Permno | Average Return |\n')
        f.write('|--------|----------------|\n')
        for _, row in top_10.iterrows():
            f.write(f"| {row['permno']} | {row['return']:.4f} |\n")
    
    print("Top 10 holdings saved to reports/top_10_holdings.md")

def extract_feature_importances():
    """Extract XGBoost feature importances (gain) and save as CSV."""
    print("Extracting XGBoost feature importances...")
    
    # Load feature names from one of the training window files
    try:
        train_data = pd.read_csv('data/splits/train_window01.csv', nrows=1)
        feature_names = [col for col in train_data.columns if col not in ['date', 'permno', 'return']]
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return
    
    # Load feature importances from model pickles
    model_files = glob.glob('/Users/BTCJULIAN/Documents/GitHub/fine_695_individual_assignment/models/xgb_window*.pkl')
    if not model_files:
        print("No model pickle files found. Skipping.")
        return
    
    # Aggregate feature importances across all windows
    all_importances = {}
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                if hasattr(model, 'feature_importances_'):
                    for feature, importance in zip(feature_names, model.feature_importances_):
                        if feature in all_importances:
                            all_importances[feature] += importance
                        else:
                            all_importances[feature] = importance
        except Exception as e:
            print(f"Error loading {model_file}: {str(e)}")
    
    if not all_importances:
        print("No feature importances found in model pickles. Skipping.")
        return
    
    # Convert to DataFrame
    feat_imp_df = pd.DataFrame(list(all_importances.items()), columns=['Feature', 'Importance'])
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
    
    # Save as CSV
    os.makedirs('reports', exist_ok=True)
    feat_imp_df.to_csv('reports/feat_imp.csv', index=False)
    print("Feature importances saved to reports/feat_imp.csv")

def map_macro_events():
    """Map macro events (2010-23) to performance spikes and save as annotated list."""
    print("Mapping macro events to performance spikes...")
    # Load portfolio returns
    portfolio_ret = pd.read_csv('portfolio_ret.csv')
    portfolio_ret['date'] = pd.to_datetime(portfolio_ret['date'])
    
    # Define macro events (example events, replace with actual events)
    macro_events = {
        '2010-05-06': 'Flash Crash',
        '2011-08-05': 'US Credit Downgrade',
        '2012-09-13': 'QE3 Announced',
        '2013-05-22': 'Taper Tantrum',
        '2014-10-15': 'Oil Price Crash',
        '2015-08-24': 'Chinese Market Crash',
        '2016-06-24': 'Brexit Vote',
        '2017-01-20': 'Trump Inauguration',
        '2018-02-05': 'Volatility Spike',
        '2019-08-14': 'Yield Curve Inversion',
        '2020-03-23': 'COVID-19 Crash',
        '2021-01-06': 'Capitol Riot',
        '2022-02-24': 'Russia-Ukraine War',
        '2023-03-10': 'SVB Collapse'
    }
    
    # Map events to portfolio returns
    events_list = []
    for date, event in macro_events.items():
        date = pd.to_datetime(date)
        if date in portfolio_ret['date'].values:
            ret = portfolio_ret[portfolio_ret['date'] == date]['portfolio_return'].values[0]
            events_list.append(f"{date.strftime('%Y-%m-%d')}: {event} - Return: {ret:.4f}")
    
    # Save as annotated list
    os.makedirs('reports', exist_ok=True)
    with open('reports/macro_events.md', 'w') as f:
        f.write('# Macro Events and Portfolio Performance\n\n')
        for event in events_list:
            f.write(f"- {event}\n")
    
    print("Macro events mapped and saved to reports/macro_events.md")

def main():
    """Main function to run all analysis tasks."""
    try:
        identify_top_holdings()
        extract_feature_importances()
        map_macro_events()
        print("Analysis tasks completed successfully.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 