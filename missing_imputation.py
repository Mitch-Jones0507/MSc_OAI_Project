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