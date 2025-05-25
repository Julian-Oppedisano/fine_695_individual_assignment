import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# Load each CSV into a DataFrame
print("Loading data files...")
mma_sample_v2 = pd.read_csv('data/raw/mma_sample_v2.csv')
factor_char_list = pd.read_csv('data/raw/factor_char_list.csv')
mkt_ind = pd.read_csv('data/raw/mkt_ind.csv')

# Convert date column to datetime with correct format
print("\n=== Date Format Investigation ===")
print("\nRaw date values (first 5 rows):")
print(mma_sample_v2['date'].head())

# Convert dates from YYYYMMDD format to datetime
mma_sample_v2['date'] = pd.to_datetime(mma_sample_v2['date'], format='%Y%m%d')

print("\nConverted date values (first 5 rows):")
print(mma_sample_v2['date'].head())

print("\nDate range:")
print(f"Start date: {mma_sample_v2['date'].min()}")
print(f"End date: {mma_sample_v2['date'].max()}")

# Create date distribution plot
plt.figure(figsize=(12, 6))
mma_sample_v2['date'].value_counts().sort_index().plot(kind='line')
plt.title('Distribution of Dates in MMA Sample V2')
plt.xlabel('Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/date_distribution.png')
plt.close()

# Basic information about each dataset
print("\n=== Basic Information ===")
print("\nMMA Sample V2 Info:")
print(mma_sample_v2.info())
print("\nFactor Characteristic List Info:")
print(factor_char_list.info())
print("\nMarket Industry Info:")
print(mkt_ind.info())

# Check for missing values
print("\n=== Missing Values Analysis ===")
print("\nMissing values in MMA Sample V2:")
missing_mma = mma_sample_v2.isnull().sum()
missing_mma_pct = (missing_mma / len(mma_sample_v2)) * 100
missing_mma_df = pd.DataFrame({
    'Missing Values': missing_mma,
    'Percentage': missing_mma_pct
})
print(missing_mma_df[missing_mma_df['Missing Values'] > 0])

print("\nMissing values in Factor Characteristic List:")
missing_factor = factor_char_list.isnull().sum()
missing_factor_pct = (missing_factor / len(factor_char_list)) * 100
missing_factor_df = pd.DataFrame({
    'Missing Values': missing_factor,
    'Percentage': missing_factor_pct
})
print(missing_factor_df[missing_factor_df['Missing Values'] > 0])

print("\nMissing values in Market Industry:")
missing_mkt = mkt_ind.isnull().sum()
missing_mkt_pct = (missing_mkt / len(mkt_ind)) * 100
missing_mkt_df = pd.DataFrame({
    'Missing Values': missing_mkt,
    'Percentage': missing_mkt_pct
})
print(missing_mkt_df[missing_mkt_df['Missing Values'] > 0])

# Create date column for mkt_ind using year and month
mkt_ind['date'] = pd.to_datetime(mkt_ind[['year', 'month']].assign(day=1))

# Print date ranges
print("\n=== Date Ranges ===")
print(f"MMA Sample V2 date range: {mma_sample_v2['date'].min()} to {mma_sample_v2['date'].max()}")
print(f"Market Industry date range: {mkt_ind['date'].min()} to {mkt_ind['date'].max()}")

# Print total unique permno values
total_unique_permno = mma_sample_v2['permno'].nunique()
print(f'\nTotal unique permno values across all months: {total_unique_permno}')

# Group by date and count unique permno
unique_counts = mma_sample_v2.groupby('date')['permno'].nunique()

# Print summary statistics
print('\nSummary of unique permno counts per month:')
print(unique_counts.describe())

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(unique_counts, bins=30, edgecolor='black')
plt.axvline(x=100, color='r', linestyle='--', label='Required minimum (100)')
plt.xlabel('Number of Unique permno per Month')
plt.ylabel('Frequency')
plt.title('Distribution of Unique permno Counts Across Months')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/permno_distribution.png')
plt.close()

# Print months with minimum and maximum counts
print('\nMonths with minimum unique permno:')
print(unique_counts[unique_counts == unique_counts.min()])
print('\nMonths with maximum unique permno:')
print(unique_counts[unique_counts == unique_counts.max()])

# Basic statistics for numeric columns in mma_sample_v2
print("\n=== Basic Statistics for MMA Sample V2 ===")
numeric_cols = mma_sample_v2.select_dtypes(include=[np.number]).columns
print(mma_sample_v2[numeric_cols].describe())

# Create correlation heatmap for selected numeric columns
print("\n=== Correlation Analysis ===")
# Select a subset of columns for correlation analysis (to avoid memory issues)
selected_cols = numeric_cols[:20]  # First 20 numeric columns
correlation_matrix = mma_sample_v2[selected_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (First 20 Numeric Features)')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png')
plt.close()

# Save summary statistics to a text file
with open('reports/data_summary.txt', 'w') as f:
    f.write("=== Data Summary Report ===\n\n")
    f.write("1. Dataset Shapes:\n")
    f.write(f"MMA Sample V2: {mma_sample_v2.shape}\n")
    f.write(f"Factor Characteristic List: {factor_char_list.shape}\n")
    f.write(f"Market Industry: {mkt_ind.shape}\n\n")
    
    f.write("2. Missing Values Summary:\n")
    f.write("MMA Sample V2:\n")
    f.write(missing_mma_df[missing_mma_df['Missing Values'] > 0].to_string())
    f.write("\n\nFactor Characteristic List:\n")
    f.write(missing_factor_df[missing_factor_df['Missing Values'] > 0].to_string())
    f.write("\n\nMarket Industry:\n")
    f.write(missing_mkt_df[missing_mkt_df['Missing Values'] > 0].to_string())
    
    f.write("\n\n3. Date Ranges:\n")
    f.write(f"MMA Sample V2: {mma_sample_v2['date'].min()} to {mma_sample_v2['date'].max()}\n")
    f.write(f"Market Industry: {mkt_ind['date'].min()} to {mkt_ind['date'].max()}\n")
    
    f.write("\n4. Unique permno Statistics:\n")
    f.write(f"Total unique permno: {total_unique_permno}\n")
    f.write("\nMonthly permno counts:\n")
    f.write(unique_counts.describe().to_string())

# Assert mma_sample_v2 contains >= 100 unique permno per month
if 'date' not in mma_sample_v2.columns or 'permno' not in mma_sample_v2.columns:
    raise ValueError('mma_sample_v2 must contain columns "date" and "permno"')

if (unique_counts < 100).any():
    failed_months = unique_counts[unique_counts < 100]
    raise AssertionError(f"Some months have fewer than 100 unique permno:\n{failed_months}")

print('\nAll CSVs loaded successfully. Each month in mma_sample_v2 has at least 100 unique permno.')
print('\nEDA completed. Summary report saved to reports/data_summary.txt')

# --- MISSING VALUE HANDLING ---
print("\n=== Missing Value Handling ===")
missing_pct = mma_sample_v2.isnull().mean() * 100
print("\nMissing percentage by column (top 20):")
print(missing_pct.sort_values(ascending=False).head(20))

# Drop columns with >50% missing values
cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
print(f"\nDropping columns with >50% missing values: {cols_to_drop}")
mma_sample_v2_clean = mma_sample_v2.drop(columns=cols_to_drop)

# Impute numeric columns with median, categorical with mode
def impute_column(col):
    if col.dtype in [np.float64, np.int64]:
        return col.fillna(col.median())
    elif col.dtype == 'O':
        return col.fillna(col.mode().iloc[0] if not col.mode().empty else 'Missing')
    else:
        return col

mma_sample_v2_clean = mma_sample_v2_clean.apply(impute_column)

# Save cleaned data
os.makedirs('data/processed', exist_ok=True)
mma_sample_v2_clean.to_csv('data/processed/mma_sample_v2_clean.csv', index=False)
print("\nSaved cleaned data to data/processed/mma_sample_v2_clean.csv")

# --- MULTICOLLINEARITY CHECKS ---
print("\n=== Multicollinearity Checks ===")
# Exclude target variable and date-related columns from correlation analysis
numeric_cols = mma_sample_v2_clean.select_dtypes(include=[np.number]).columns.drop(['permno', 'date', 'stock_exret'], errors='ignore')
data_numeric = mma_sample_v2_clean[numeric_cols]
corr_matrix = data_numeric.corr().abs()
threshold = 0.9
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print(f"\nHighly correlated pairs (|corr| > {threshold}):")
for a, b, corr in sorted(high_corr_pairs, key=lambda x: -x[2]):
    print(f"{a} <-> {b}: {corr:.3f}")

features_to_drop = set()
for a, b, corr in high_corr_pairs:
    mean_corr_a = corr_matrix[a].mean()
    mean_corr_b = corr_matrix[b].mean()
    drop = a if mean_corr_a > mean_corr_b else b
    features_to_drop.add(drop)

print(f"\nDropping features due to high multicollinearity: {sorted(features_to_drop)}")
data_nocollinear = mma_sample_v2_clean.drop(columns=list(features_to_drop))
data_nocollinear.to_csv('data/processed/mma_sample_v2_nocollinear.csv', index=False)
print("\nSaved reduced dataset to data/processed/mma_sample_v2_nocollinear.csv")

# --- TARGET VARIABLE ANALYSIS ---
print("\n=== Target Variable Analysis (stock_exret) ===")

# Note about data issue
print("\nNote: The ret_eom column appears to be incorrectly populated with date values.")
print("Using stock_exret as the target variable instead, which contains the actual returns.")

# Basic statistics
print("\nBasic statistics of stock_exret:")
print(data_nocollinear['stock_exret'].describe())

# Create directory for target analysis plots
os.makedirs('figures/target_analysis', exist_ok=True)

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=data_nocollinear, x='stock_exret', bins=50)
plt.title('Distribution of Stock Excess Returns')
plt.xlabel('Excess Return')
plt.ylabel('Count')
plt.savefig('figures/target_analysis/return_distribution.png')
plt.close()

# Time series plot of average monthly returns
monthly_returns = data_nocollinear.groupby('date')['stock_exret'].mean()
plt.figure(figsize=(12, 6))
monthly_returns.plot()
plt.title('Average Monthly Excess Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Average Excess Return')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/target_analysis/monthly_returns.png')
plt.close()

# Box plot of returns by year
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_nocollinear, x='year', y='stock_exret')
plt.title('Distribution of Excess Returns by Year')
plt.xlabel('Year')
plt.ylabel('Excess Return')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/target_analysis/returns_by_year.png')
plt.close()

# Outlier analysis using IQR method
Q1 = data_nocollinear['stock_exret'].quantile(0.25)
Q3 = data_nocollinear['stock_exret'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data_nocollinear[(data_nocollinear['stock_exret'] < lower_bound) | 
                           (data_nocollinear['stock_exret'] > upper_bound)]

print("\nOutlier Analysis:")
print(f"Number of outliers: {len(outliers)}")
print(f"Percentage of outliers: {(len(outliers) / len(data_nocollinear)) * 100:.2f}%")
print(f"Lower bound: {lower_bound:.4f}")
print(f"Upper bound: {upper_bound:.4f}")

# Save target analysis summary
with open('reports/target_analysis.txt', 'w') as f:
    f.write("=== Target Variable Analysis Summary ===\n\n")
    f.write("Note: The ret_eom column appears to be incorrectly populated with date values.\n")
    f.write("Using stock_exret as the target variable instead, which contains the actual returns.\n\n")
    f.write("Basic Statistics:\n")
    f.write(data_nocollinear['stock_exret'].describe().to_string())
    f.write("\n\nOutlier Analysis:\n")
    f.write(f"Number of outliers: {len(outliers)}\n")
    f.write(f"Percentage of outliers: {(len(outliers) / len(data_nocollinear)) * 100:.2f}%\n")
    f.write(f"Lower bound: {lower_bound:.4f}\n")
    f.write(f"Upper bound: {upper_bound:.4f}\n")

print("\nTarget analysis completed. Results saved to reports/target_analysis.txt")

# --- FEATURE DISTRIBUTION ANALYSIS ---
print("\n=== Feature Distribution Analysis ===")

os.makedirs('figures/feature_distributions', exist_ok=True)
feature_dist_report = []
numeric_features = data_nocollinear.select_dtypes(include=[np.number]).columns.drop(['permno', 'date', 'stock_exret'], errors='ignore')

for col in numeric_features:
    desc = data_nocollinear[col].describe()
    skewness = data_nocollinear[col].skew()
    Q1 = desc['25%']
    Q3 = desc['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((data_nocollinear[col] < lower_bound) | (data_nocollinear[col] > upper_bound)).sum()
    outlier_pct = 100 * outliers / len(data_nocollinear)
    feature_dist_report.append(f"Feature: {col}\nSummary: {desc.to_string()}\nSkewness: {skewness:.2f}\nOutliers: {outliers} ({outlier_pct:.2f}%)\nLower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}\n")
    # Plot histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(data_nocollinear[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'figures/feature_distributions/{col}_hist.png')
    plt.close()

with open('reports/feature_distribution_analysis.txt', 'w') as f:
    f.write("=== Feature Distribution Analysis ===\n\n")
    for entry in feature_dist_report:
        f.write(entry + '\n')

print("\nFeature distribution analysis completed. Results saved to reports/feature_distribution_analysis.txt")

# --- TIME SERIES CHECKS ---
print("\n=== Time Series Checks ===")
os.makedirs('figures/time_series_analysis', exist_ok=True)
time_series_report = []

dates = pd.to_datetime(data_nocollinear['date'].unique())
dates = pd.Series(sorted(dates))
full_range = pd.date_range(start=dates.min(), end=dates.max(), freq='M')
missing_months = full_range.difference(dates)

time_series_report.append(f"Date range: {dates.min().date()} to {dates.max().date()}")
time_series_report.append(f"Total months in range: {len(full_range)}")
time_series_report.append(f"Months present in data: {len(dates)}")
time_series_report.append(f"Missing months: {len(missing_months)}")
if len(missing_months) > 0:
    time_series_report.append(f"List of missing months: {[d.strftime('%Y-%m') for d in missing_months]}")
else:
    time_series_report.append("No missing months in the time series.")

# Seasonality: average returns by calendar month
monthly_avg = data_nocollinear.copy()
monthly_avg['month'] = monthly_avg['date'].dt.month
month_means = monthly_avg.groupby('month')['stock_exret'].mean()
plt.figure(figsize=(8, 4))
month_means.plot(kind='bar')
plt.title('Average End-of-Month Return by Calendar Month')
plt.xlabel('Month')
plt.ylabel('Average Return')
plt.tight_layout()
plt.savefig('figures/time_series_analysis/avg_return_by_month.png')
plt.close()
time_series_report.append("\nAverage return by calendar month:")
time_series_report.append(month_means.to_string())

# Seasonality: average returns by year
monthly_avg['year'] = monthly_avg['date'].dt.year
year_means = monthly_avg.groupby('year')['stock_exret'].mean()
plt.figure(figsize=(10, 4))
year_means.plot()
plt.title('Average End-of-Month Return by Year')
plt.xlabel('Year')
plt.ylabel('Average Return')
plt.tight_layout()
plt.savefig('figures/time_series_analysis/avg_return_by_year.png')
plt.close()
time_series_report.append("\nAverage return by year:")
time_series_report.append(year_means.to_string())

with open('reports/time_series_analysis.txt', 'w') as f:
    f.write("=== Time Series Analysis ===\n\n")
    for entry in time_series_report:
        f.write(entry + '\n')

print("\nTime series analysis completed. Results saved to reports/time_series_analysis.txt")

# --- TARGET TRANSFORMATION ANALYSIS ---
print("\n=== Target Transformation Analysis ===")
os.makedirs('figures/target_transformation', exist_ok=True)
transformation_report = []

# Original distribution statistics
original_skew = stats.skew(data_nocollinear['stock_exret'])
original_kurt = stats.kurtosis(data_nocollinear['stock_exret'])
shapiro_test = stats.shapiro(data_nocollinear['stock_exret'])

transformation_report.append("Original Distribution Statistics:")
transformation_report.append(f"Skewness: {original_skew:.4f}")
transformation_report.append(f"Kurtosis: {original_kurt:.4f}")
transformation_report.append(f"Shapiro-Wilk Test: W={shapiro_test[0]:.4f}, p-value={shapiro_test[1]:.4e}")
transformation_report.append("Interpretation: p-value < 0.05 indicates non-normal distribution")

# Q-Q plot of original data
plt.figure(figsize=(8, 6))
stats.probplot(data_nocollinear['stock_exret'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Original Returns')
plt.savefig('figures/target_transformation/qq_plot_original.png')
plt.close()

# Try different transformations
transformations = {
    'yeo_johnson': PowerTransformer(method='yeo-johnson').fit_transform(data_nocollinear[['stock_exret']]).ravel(),
    'winsorized': stats.mstats.winsorize(data_nocollinear['stock_exret'], limits=[0.025, 0.025]),
    'rank': stats.rankdata(data_nocollinear['stock_exret']) / len(data_nocollinear['stock_exret'])
}

# Compare transformations
transformation_report.append("\nTransformation Results:")
for name, transformed in transformations.items():
    skew = stats.skew(transformed)
    kurt = stats.kurtosis(transformed)
    shapiro = stats.shapiro(transformed)
    
    transformation_report.append(f"\n{name.upper()} Transformation:")
    transformation_report.append(f"Skewness: {skew:.4f}")
    transformation_report.append(f"Kurtosis: {kurt:.4f}")
    transformation_report.append(f"Shapiro-Wilk Test: W={shapiro[0]:.4f}, p-value={shapiro[1]:.4e}")
    
    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(transformed, kde=True)
    plt.title(f'Distribution of {name.upper()} Transformed Returns')
    plt.savefig(f'figures/target_transformation/{name}_distribution.png')
    plt.close()
    
    # Q-Q plot
    plt.figure(figsize=(8, 6))
    stats.probplot(transformed, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {name.upper()} Transformed Returns')
    plt.savefig(f'figures/target_transformation/qq_plot_{name}.png')
    plt.close()

# Save transformation analysis report
with open('reports/target_transformation_analysis.txt', 'w') as f:
    f.write("=== Target Transformation Analysis ===\n\n")
    for entry in transformation_report:
        f.write(entry + '\n')

print("\nTarget transformation analysis completed. Results saved to reports/target_transformation_analysis.txt") 