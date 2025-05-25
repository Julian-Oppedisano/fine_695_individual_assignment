import pandas as pd
import os

# Load cleaned data
input_path = 'data/processed/mma_sample_v2_clean.csv'
df = pd.read_csv(input_path)

# Identify columns to lag: all predictors (exclude identifiers and target)
exclude_cols = ['date', 'permno', 'stock_exret', 'shrcd', 'exchcd', 'stock_ticker', 'cusip', 'comp_name', 'year', 'month', 'rf']
predictor_cols = [col for col in df.columns if col not in exclude_cols]

# Sort by permno and date to ensure correct lagging
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.sort_values(['permno', 'date'])

# Lag predictors by 1 month for each stock
for col in predictor_cols:
    df[col + '_lag1'] = df.groupby('permno')[col].shift(1)

# Drop original (unlagged) predictor columns
df_lagged = df.drop(columns=predictor_cols)

# Drop rows with any missing lagged predictors (first month for each stock)
df_lagged = df_lagged.dropna(subset=[col + '_lag1' for col in predictor_cols])

# Save lagged dataset
os.makedirs('data/processed', exist_ok=True)
df_lagged.to_csv('data/processed/mma_sample_v2_lagged.csv', index=False)

print('Lagged predictors created and saved to data/processed/mma_sample_v2_lagged.csv')

# --- Cross-sectional Z-scoring ---
print('Starting cross-sectional z-scoring of lagged predictors...')

# Identify lagged predictor columns
lagged_predictor_cols = [col for col in df_lagged.columns if col.endswith('_lag1')]

# Only use numeric lagged predictors for z-scoring
numeric_lagged_cols = df_lagged[lagged_predictor_cols].select_dtypes(include='number').columns.tolist()

# Z-score within each date (cross-sectionally)
def zscore(group):
    return (group - group.mean()) / group.std(ddof=0)

zscored = df_lagged.copy()
if numeric_lagged_cols:
    zscored[numeric_lagged_cols] = zscored.groupby('date')[numeric_lagged_cols].transform(zscore)
else:
    print('No numeric lagged predictors found for z-scoring.')

# Save z-scored dataset
zscored_path = 'data/processed/mma_sample_v2_zscored.csv'
zscored.to_csv(zscored_path, index=False)
print(f'Z-scored predictors saved to {zscored_path}')

# --- Impute remaining NaNs with cross-sectional medians ---
print('Imputing remaining missing values with cross-sectional medians...')

# Get all numeric columns (including lagged and z-scored)
numeric_cols = zscored.select_dtypes(include='number').columns.tolist()

# Impute with cross-sectional medians
for col in numeric_cols:
    zscored[col] = zscored.groupby('date')[col].transform(lambda x: x.fillna(x.median()))
    # Fill any remaining NaNs with the global median
    zscored[col] = zscored[col].fillna(zscored[col].median())
    # As a last resort, fill any remaining NaNs with zero
    zscored[col] = zscored[col].fillna(0)

# Verify no NaNs remain
assert zscored[numeric_cols].isna().sum().sum() == 0, "Some NaNs remain after imputation"

# Save final dataset
final_path = 'data/processed/mma_sample_v2_final.csv'
zscored.to_csv(final_path, index=False)
print(f'Final dataset with imputed values saved to {final_path}') 