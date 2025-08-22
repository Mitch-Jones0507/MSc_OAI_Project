import pandas as pd
from missing_imputation import moaks_shared_kl_drop_df, moaks_shared_kl_median_df
from modules.variable_analysis import run_linear_regression, run_multiple_pls, run_lasso_regression, \
 create_transformed_features
from scripts.mbm.kl_oriented.mbm_var_relationships_kl import v00_moaks_shared_kl_drop_mbm_df, v01_moaks_shared_kl_drop_mbm_df, \
 v00_moaks_shared_kl_drop_mbm_reduced_df, v01_moaks_shared_kl_drop_mbm_reduced_df

v01_kl_drop = moaks_shared_kl_drop_df['V01XRKL']
v03_kl_drop = moaks_shared_kl_drop_df['V03XRKL']
Δkl_drop = v03_kl_drop - v01_kl_drop

v01_kl_median = moaks_shared_kl_median_df['V01XRKL']
v03_kl_median = moaks_shared_kl_median_df['V03XRKL']
Δkl_median = v03_kl_median - v01_kl_median

_, v00_v01_kl_drop_results, v00_v01_kl_drop_coef_df = run_linear_regression(v00_moaks_shared_kl_drop_mbm_df,v01_kl_drop)
v00_v01_kl_drop_results = pd.DataFrame.from_dict(v00_v01_kl_drop_results, orient='index', columns=['Value'])
_, v00_v03_kl_drop_results, v00_v03_kl_drop_coef_df = run_linear_regression(v00_moaks_shared_kl_drop_mbm_df,v03_kl_drop)
v00_v03_kl_drop_results = pd.DataFrame.from_dict(v00_v03_kl_drop_results, orient='index', columns=['Value'])
_, v01_v03_kl_drop_results, v01_v03_kl_drop_coef_df = run_linear_regression(v01_moaks_shared_kl_drop_mbm_df,v03_kl_drop)
v01_v03_kl_drop_results = pd.DataFrame.from_dict(v01_v03_kl_drop_results, orient='index', columns=['Value'])
_, v00_kl_drop_change_results, v00_kl_drop_change_coef_df = run_linear_regression(v00_moaks_shared_kl_drop_mbm_df,Δkl_drop)
_, v01_kl_drop_change_results, v01_kl_drop_change_coef_df = run_linear_regression(v01_moaks_shared_kl_drop_mbm_df,Δkl_drop)

_, v00_v01_kl_drop_reduced_results, v00_v01_kl_drop_reduced_coef_df = run_linear_regression(v00_moaks_shared_kl_drop_mbm_reduced_df,v01_kl_drop)
_, v00_v03_kl_drop_reduced_results, v00_v03_kl_drop_reduced_coef_df = run_linear_regression(v00_moaks_shared_kl_drop_mbm_reduced_df,v03_kl_drop)
_, v01_v03_kl_drop_reduced_results, v01_v03_kl_drop_reduced_coef_df = run_linear_regression(v01_moaks_shared_kl_drop_mbm_reduced_df,v03_kl_drop)
_, v00_kl_drop_reduced_change_results, v00_kl_drop_reduced_change_coef_df = run_linear_regression(v00_moaks_shared_kl_drop_mbm_reduced_df,Δkl_drop)
_, v01_kl_drop_reduced_change_results, v01_kl_drop_reduced_change_coef_df = run_linear_regression(v01_moaks_shared_kl_drop_mbm_reduced_df,Δkl_drop)

_, v00_v01_kl_drop_pls_results, v00_v01_kl_drop_pls_coef_df, v00_v01_kl_drop_pls_constructs = run_multiple_pls(v00_moaks_shared_kl_drop_mbm_df, v01_kl_drop)
v00_v01_kl_drop_pls_results = pd.DataFrame(list(v00_v01_kl_drop_pls_results.items()), columns=['Feature', 'Value'])
_, v00_v03_kl_drop_pls_results, v00_v03_kl_drop_pls_coef_df, v00_v03_kl_drop_pls_constructs = run_multiple_pls(v00_moaks_shared_kl_drop_mbm_df, v03_kl_drop)
v00_v03_kl_drop_pls_results = pd.DataFrame(list(v00_v03_kl_drop_pls_results.items()), columns=['Feature', 'Value'])
_, v01_v03_kl_drop_pls_results, v01_v03_kl_drop_pls_coef_df, v01_v03_kl_drop_pls_constructs = run_multiple_pls(v01_moaks_shared_kl_drop_mbm_df, v03_kl_drop)
v01_v03_kl_drop_pls_results = pd.DataFrame(list(v01_v03_kl_drop_pls_results.items()), columns=['Feature', 'Value'])
_, v00_kl_drop_change_pls_results, v00_kl_drop_pls_change_coef_df, v00_kl_drop_change_pls_constructs  = run_multiple_pls(v00_moaks_shared_kl_drop_mbm_df, Δkl_drop)
_, v01_kl_drop_change_pls_results, v01_kl_drop_pls_change_coef_df, v01_kl_drop_change_pls_constructs= run_multiple_pls(v01_moaks_shared_kl_drop_mbm_df, Δkl_drop)

_, v00_v01_kl_drop_lasso_results, v00_v01_kl_drop_lasso_coef_df = run_lasso_regression(v00_moaks_shared_kl_drop_mbm_df,v01_kl_drop)
_, v00_v03_kl_drop_lasso_results, v00_v03_kl_drop_lasso_coef_df = run_lasso_regression(v00_moaks_shared_kl_drop_mbm_df,v03_kl_drop)
_, v01_v03_kl_drop_lasso_results, v01_v03_kl_drop_lasso_coef_df = run_lasso_regression(v01_moaks_shared_kl_drop_mbm_df,v03_kl_drop)
_, v00_kl_drop_change_lasso_results, v00_kl_drop_change_lasso_coef_df = run_lasso_regression(v00_moaks_shared_kl_drop_mbm_df,Δkl_drop)
_, v01_kl_drop_change_lasso_results, v01_kl_drop_change_lasso_coef_df = run_lasso_regression(v01_moaks_shared_kl_drop_mbm_df,Δkl_drop)

v00_nonzero_features = v00_v01_kl_drop_lasso_coef_df[
    (v00_v01_kl_drop_lasso_coef_df['Coefficient'] > 1e-9) &
    (v00_v01_kl_drop_lasso_coef_df['Feature'] != 'Intercept')
]['Feature'].drop_duplicates().tolist()

v00_kl_drop_mbm_columns_filtered = v00_moaks_shared_kl_drop_mbm_df[v00_nonzero_features]
v00_kl_drop_mbm_columns_transformed = create_transformed_features(v00_kl_drop_mbm_columns_filtered)

v01_nonzero_features = v01_v03_kl_drop_lasso_coef_df[
    (v01_v03_kl_drop_lasso_coef_df['Coefficient'] > 1e-9) &
    (v01_v03_kl_drop_lasso_coef_df['Feature'] != 'Intercept')
]['Feature'].drop_duplicates().tolist()

v01_kl_drop_mbm_columns_filtered = v01_moaks_shared_kl_drop_mbm_df[v01_nonzero_features]
v01_kl_drop_mbm_columns_transformed = create_transformed_features(v01_kl_drop_mbm_columns_filtered)

_, v00_v01_kl_left_drop_transformed_lasso_results, v00_v01_kl_left_transformed_lasso_coef_matrix = run_lasso_regression(v00_kl_drop_mbm_columns_transformed, v01_kl_drop)
_, v00_v03_kl_left_drop_transformed_lasso_results, v00_v03_kl_left_transformed_lasso_coef_matrix = run_lasso_regression(v00_kl_drop_mbm_columns_transformed, v03_kl_drop)
_, v01_v03_kl_left_drop_transformed_lasso_results, v01_v03_kl_left_transformed_lasso_coef_matrix = run_lasso_regression(v01_kl_drop_mbm_columns_transformed, v03_kl_drop)

