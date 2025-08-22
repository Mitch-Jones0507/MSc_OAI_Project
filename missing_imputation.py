import pandas as pd
from config import moaks_shared_kl_df, moaks_shared_womac_df

moaks_shared_kl_drop_df = moaks_shared_kl_df.dropna()
moaks_shared_womac_drop_df = moaks_shared_womac_df.dropna()

moaks_shared_kl_median_df = moaks_shared_kl_df.copy()
moaks_shared_womac_median_df = moaks_shared_womac_df.copy()

moaks_shared_kl_numeric_columns = moaks_shared_kl_median_df.drop(columns=['ID','SIDE']).select_dtypes(include='number')
moaks_shared_womac_numeric_columns = moaks_shared_womac_median_df.drop(columns=['ID','SIDE']).select_dtypes(include='number')

for column in moaks_shared_kl_numeric_columns.columns:
    moaks_shared_kl_median_df[column] = moaks_shared_kl_median_df[column].fillna(moaks_shared_kl_median_df[column].median())

for column in moaks_shared_womac_numeric_columns.columns:
    moaks_shared_womac_median_df[column] = moaks_shared_womac_median_df[column].fillna(moaks_shared_womac_median_df[column].median())

moaks_shared_kl_median_table = moaks_shared_kl_numeric_columns.median().reset_index()
moaks_shared_kl_median_table.columns = ['Variable','Median']
moaks_shared_kl_median_table = moaks_shared_kl_median_table.sort_values(by=['Median'], ascending=False).T

moaks_shared_womac_median_table = moaks_shared_womac_numeric_columns.median().reset_index()
moaks_shared_womac_median_table.columns = ['Variable','Median']
moaks_shared_womac_median_table = moaks_shared_womac_median_table.sort_values(by=['Median'], ascending=False).T

moaks_shared_kl_drop_frequency = pd.DataFrame({
    variable: moaks_shared_kl_drop_df[variable].value_counts().reindex(range(0,5),fill_value=0)
    for variable in ['V01XRKL','V03XRKL']
})

moaks_mbm_df = moaks_shared_kl_df.filter(like='MBM')

# df = your 45-variable dataframe
bml_frequency_table = pd.DataFrame({
    col: moaks_mbm_df[col].value_counts().reindex([0,1,2,3,4], fill_value=0)
    for col in moaks_mbm_df.columns
}).T  # transpose so rows = variables

bml_frequency_table.index.name = 'Variable'
bml_frequency_table.columns.name = 'Score'

