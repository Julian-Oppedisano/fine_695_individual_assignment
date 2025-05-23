import pandas as pd

# Load each CSV into a DataFrame
mma_sample_v2 = pd.read_csv('data/raw/mma_sample_v2.csv')
factor_char_list = pd.read_csv('data/raw/factor_char_list.csv')
mkt_ind = pd.read_csv('data/raw/mkt_ind.csv')

print('All CSVs loaded successfully.') 