
from missing_imputation import moaks_shared_womac_drop_df, moaks_shared_womac_median_df
from modules.variable_analysis import run_multivariate_linear_regression

moaks_shared_womac_drop_left_df = moaks_shared_womac_drop_df[moaks_shared_womac_drop_df['SIDE']==1]
moaks_shared_womac_drop_right_df = moaks_shared_womac_drop_df[moaks_shared_womac_drop_df['SIDE']==2]

v00_womac_drop_left_mbm_columns = moaks_shared_womac_drop_left_df[[column for column in moaks_shared_womac_drop_left_df.columns if 'V00MBM' in column]]
v01_womac_drop_left_mbm_columns = moaks_shared_womac_drop_left_df[[column for column in moaks_shared_womac_drop_left_df.columns if 'V01MBM' in column]]

v00_womac_drop_right_mbm_columns = moaks_shared_womac_drop_right_df[[column for column in moaks_shared_womac_drop_right_df.columns if 'V00MBM' in column]]
v01_womac_drop_right_mbm_columns = moaks_shared_womac_drop_right_df[[column for column in moaks_shared_womac_drop_right_df.columns if 'V01MBM' in column]]

v01_womac_drop_left_womac_columns = moaks_shared_womac_drop_left_df[['V01WOMADLL','V01WOMKPL','V01WOMSTFL']]
v03_womac_drop_left_womac_columns = moaks_shared_womac_drop_left_df[['V03WOMADLL','V03WOMKPL','V03WOMSTFL']]

v01_womac_drop_right_womac_columns = moaks_shared_womac_drop_right_df[['V01WOMADLR','V01WOMKPR','V01WOMSTFR']]
v03_womac_drop_right_womac_columns = moaks_shared_womac_drop_right_df[['V03WOMADLR','V03WOMKPR','V03WOMSTFR']]

_, v00_v01_womac_drop_results, v00_v01_womac_p_matrix = run_multivariate_linear_regression(v00_womac_drop_left_mbm_columns,v01_womac_drop_left_womac_columns)
_, v00_v03_womac_drop_results, v00_v03_womac_p_matrix = run_multivariate_linear_regression(v00_womac_drop_left_mbm_columns,v03_womac_drop_left_womac_columns)
_, v01_v03_womac_drop_results, v01_v03_womac_p_matrix = run_multivariate_linear_regression(v01_womac_drop_left_mbm_columns,v03_womac_drop_left_womac_columns)
