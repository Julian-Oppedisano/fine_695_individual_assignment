import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import datetime
import os

def load_data() -> pd.DataFrame:
    """Load the final processed dataset."""
    return pd.read_csv('data/processed/mma_sample_v2_final.csv')

def create_expanding_windows(
    df: pd.DataFrame,
    initial_train_end: str = '2007-12-31',
    val_window_size: int = 24,  # 2 years in months
    test_window_size: int = 12,  # 1 year in months
    step_size: int = 12  # Move forward 1 year at a time
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create expanding windows for time series cross-validation.
    
    Args:
        df: DataFrame with 'date' column
        initial_train_end: Last month of initial training period
        val_window_size: Size of validation window in months
        test_window_size: Size of test window in months
        step_size: Number of months to move forward each iteration
    
    Returns:
        List of (train, validation, test) DataFrame tuples
    """
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get unique dates
    dates = df['date'].unique()
    dates = np.sort(dates)
    
    # Find initial train end index
    initial_train_end = pd.to_datetime(initial_train_end)
    train_end_idx = np.where(dates == initial_train_end)[0][0]
    
    windows = []
    current_idx = train_end_idx
    
    while current_idx + val_window_size + test_window_size < len(dates):
        # Get window indices
        train_end = current_idx
        val_end = train_end + val_window_size
        test_end = val_end + test_window_size
        
        # Create windows
        train_dates = dates[:train_end + 1]
        val_dates = dates[train_end + 1:val_end + 1]
        test_dates = dates[val_end + 1:test_end + 1]
        
        # Create DataFrames
        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]
        test_df = df[df['date'].isin(test_dates)]
        
        windows.append((train_df, val_df, test_df))
        
        # Move forward
        current_idx += step_size
    
    return windows

def verify_windows(windows: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> None:
    """Verify the first set of windows matches the required specifications."""
    train_df, val_df, test_df = windows[0]
    
    # Verify first train window
    train_start = train_df['date'].min()
    train_end = train_df['date'].max()
    assert train_start.year == 2000 and train_start.month == 2 and train_start.day == 29, \
        f"First train window should start at 2000-02-29, got {train_start}"
    assert train_end.year == 2007 and train_end.month == 12, \
        f"First train window should end at 2007-12, got {train_end}"
    
    # Verify first validation window
    val_start = val_df['date'].min()
    val_end = val_df['date'].max()
    assert val_start.year == 2008 and val_start.month == 1, \
        f"First validation window should start at 2008-01, got {val_start}"
    assert val_end.year == 2009 and val_end.month == 12, \
        f"First validation window should end at 2009-12, got {val_end}"
    
    # Verify first test window
    test_start = test_df['date'].min()
    test_end = test_df['date'].max()
    assert test_start.year == 2010 and test_start.month == 1, \
        f"First test window should start at 2010-01, got {test_start}"
    assert test_end.year == 2010 and test_end.month == 12, \
        f"First test window should end at 2010-12, got {test_end}"
    
    print("All window verifications passed!")

def main():
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Create windows
    print("Creating expanding windows...")
    windows = create_expanding_windows(df)
    
    # Verify windows
    print("Verifying windows...")
    verify_windows(windows)
    
    # Print summary
    print(f"\nCreated {len(windows)} expanding windows")
    print("Window sizes:")
    for i, (train, val, test) in enumerate(windows):
        print(f"\nWindow {i+1}:")
        print(f"Train: {train['date'].min()} to {train['date'].max()} ({len(train)} rows)")
        print(f"Validation: {val['date'].min()} to {val['date'].max()} ({len(val)} rows)")
        print(f"Test: {test['date'].min()} to {test['date'].max()} ({len(test)} rows)")
    
    # Save all splits
    os.makedirs('data/splits', exist_ok=True)
    for i, (train, val, test) in enumerate(windows, 1):
        train.to_csv(f'data/splits/train_window{i:02d}.csv', index=False)
        val.to_csv(f'data/splits/val_window{i:02d}.csv', index=False)
        test.to_csv(f'data/splits/test_window{i:02d}.csv', index=False)
    print(f"\nSaved all splits to data/splits/")

if __name__ == "__main__":
    main() 