import pandas as pd
from modules.machine_learning import run_xgb_multioutput_regressor
from scripts.mbm.womac_oriented.mbm_var_relationships_womac import v00_moaks_shared_womac_drop_mbm_df, v01_moaks_shared_womac_drop_mbm_df
from scripts.mbm.womac_oriented.womac_mbm_relationships import v00_womac_drop_left_mbm_columns, v01_womac_drop_left_mbm_columns, \
    v00_womac_drop_right_mbm_columns, v01_womac_drop_right_mbm_columns, v00_v01_womac_left_lasso_coef_matrix, v00_v03_womac_left_lasso_coef_matrix, v01_v03_womac_left_lasso_coef_matrix, \
    v00_v01_womac_right_lasso_coef_matrix, v00_v03_womac_right_lasso_coef_matrix, v01_v03_womac_right_lasso_coef_matrix, \
    v00_womac_left_change_lasso_coef_matrix, v01_womac_left_change_lasso_coef_matrix, v01_womac_right_change_lasso_coef_matrix, \
    v01_womac_right_drop_change_lasso_results, v01_womac_drop_left_womac_columns, v03_womac_drop_left_womac_columns, \
    v01_womac_drop_right_womac_columns, v03_womac_drop_right_womac_columns

v00_womac_drop_left_mbm_columns = v00_womac_drop_left_mbm_columns.copy()
v01_womac_drop_left_mbm_columns = v01_womac_drop_left_mbm_columns.copy()
v00_womac_drop_right_mbm_columns = v00_womac_drop_right_mbm_columns.copy()
v01_womac_drop_right_mbm_columns = v01_womac_drop_right_mbm_columns.copy()

v00_v01_moaks_left_lasso_nonzero_features = v00_v01_womac_left_lasso_coef_matrix.loc[
    (v00_v01_womac_left_lasso_coef_matrix['Coefficient'].abs() > 1e-9) &
    (v00_v01_womac_left_lasso_coef_matrix['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v00_v01_moaks_shared_womac_drop_left_mbm_lasso_df = v00_womac_drop_left_mbm_columns[v00_v01_moaks_left_lasso_nonzero_features]

v00_v03_moaks_left_lasso_nonzero_features = v00_v03_womac_left_lasso_coef_matrix.loc[
    (v00_v03_womac_left_lasso_coef_matrix['Coefficient'].abs() > 1e-9) &
    (v00_v03_womac_left_lasso_coef_matrix['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v00_v03_moaks_shared_womac_drop_left_mbm_lasso_df = v00_womac_drop_left_mbm_columns[v00_v03_moaks_left_lasso_nonzero_features]

v01_v03_moaks_left_lasso_nonzero_features = v01_v03_womac_left_lasso_coef_matrix.loc[
    (v01_v03_womac_left_lasso_coef_matrix['Coefficient'].abs() > 1e-9) &
    (v01_v03_womac_left_lasso_coef_matrix['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v01_v03_moaks_shared_womac_drop_left_mbm_lasso_df = v01_womac_drop_left_mbm_columns[v01_v03_moaks_left_lasso_nonzero_features]

v00_v01_moaks_right_lasso_nonzero_features = v00_v01_womac_right_lasso_coef_matrix.loc[
    (v00_v01_womac_right_lasso_coef_matrix['Coefficient'].abs() > 1e-9) &
    (v00_v01_womac_right_lasso_coef_matrix['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v00_v01_moaks_shared_womac_drop_right_mbm_lasso_df = v00_womac_drop_right_mbm_columns[v00_v01_moaks_right_lasso_nonzero_features]

v00_v03_moaks_right_lasso_nonzero_features = v00_v03_womac_right_lasso_coef_matrix.loc[
    (v00_v03_womac_right_lasso_coef_matrix['Coefficient'].abs() > 1e-9) &
    (v00_v03_womac_right_lasso_coef_matrix['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v00_v03_moaks_shared_womac_drop_right_mbm_lasso_df = v00_womac_drop_right_mbm_columns[v00_v03_moaks_right_lasso_nonzero_features]

v01_v03_moaks_right_lasso_nonzero_features = v01_v03_womac_right_lasso_coef_matrix.loc[
    (v01_v03_womac_right_lasso_coef_matrix['Coefficient'].abs() > 1e-9) &
    (v01_v03_womac_right_lasso_coef_matrix['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v01_v03_moaks_shared_womac_drop_right_mbm_lasso_df = v01_womac_drop_right_mbm_columns[v01_v03_moaks_right_lasso_nonzero_features]

v00_v01_moaks_womac_left_model, v00_v01_moaks_womac_left_metrics, v00_v01_moaks_womac_left_coefs_df = run_xgb_multioutput_regressor(v00_womac_drop_left_mbm_columns, v01_womac_drop_left_womac_columns)
v00_v03_moaks_womac_left_model, v00_v03_moaks_womac_left_metrics, v00_v03_moaks_womac_left_coefs_df = run_xgb_multioutput_regressor(v00_womac_drop_left_mbm_columns, v03_womac_drop_left_womac_columns)
v01_v03_moaks_womac_left_model, v01_v03_moaks_womac_left_metrics, v01_v03_moaks_womac_left_coefs_df = run_xgb_multioutput_regressor(v01_womac_drop_left_mbm_columns, v03_womac_drop_left_womac_columns)

v00_v01_moaks_womac_right_model, v00_v01_moaks_womac_right_metrics, v00_v01_moaks_womac_right_coefs_df = run_xgb_multioutput_regressor(v00_womac_drop_right_mbm_columns, v01_womac_drop_right_womac_columns)
v00_v03_moaks_womac_right_model, v00_v03_moaks_womac_right_metrics, v00_v03_moaks_womac_right_coefs_df = run_xgb_multioutput_regressor(v00_womac_drop_right_mbm_columns, v03_womac_drop_right_womac_columns)
v01_v03_moaks_womac_right_model, v01_v03_moaks_womac_right_metrics, v01_v03_moaks_womac_right_coefs_df = run_xgb_multioutput_regressor(v01_womac_drop_right_mbm_columns, v03_womac_drop_right_womac_columns)

v00_v01_moaks_womac_left_lasso_model, v00_v01_moaks_womac_left_lasso_metrics, v00_v01_moaks_womac_left_lasso_coefs_df = run_xgb_multioutput_regressor(v00_v01_moaks_shared_womac_drop_left_mbm_lasso_df.loc[:, ~v00_v01_moaks_shared_womac_drop_left_mbm_lasso_df.columns.duplicated()], v01_womac_drop_left_womac_columns)

v00_v03_moaks_womac_left_lasso_model, v00_v03_moaks_womac_left_lasso_metrics, v00_v03_moaks_womac_left_lasso_coefs_df = run_xgb_multioutput_regressor(v00_v03_moaks_shared_womac_drop_left_mbm_lasso_df.loc[:, ~v00_v03_moaks_shared_womac_drop_left_mbm_lasso_df.columns.duplicated()], v03_womac_drop_left_womac_columns)
v01_v03_moaks_womac_left_lasso_model, v01_v03_moaks_womac_left_lasso_metrics, v01_v03_moaks_womac_left_lasso_coefs_df = run_xgb_multioutput_regressor(v01_v03_moaks_shared_womac_drop_left_mbm_lasso_df.loc[:, ~v01_v03_moaks_shared_womac_drop_left_mbm_lasso_df.columns.duplicated()], v03_womac_drop_left_womac_columns)

v00_v01_moaks_womac_right_lasso_model, v00_v01_moaks_womac_right_lasso_metrics, v00_v01_moaks_womac_right_lasso_coefs_df = run_xgb_multioutput_regressor(v00_v01_moaks_shared_womac_drop_right_mbm_lasso_df.loc[:, ~v00_v01_moaks_shared_womac_drop_right_mbm_lasso_df.columns.duplicated()], v01_womac_drop_right_womac_columns)

v00_v03_moaks_womac_right_lasso_model, v00_v03_moaks_womac_right_lasso_metrics, v00_v03_moaks_womac_right_lasso_coefs_df = run_xgb_multioutput_regressor(v00_v03_moaks_shared_womac_drop_right_mbm_lasso_df.loc[:, ~v00_v03_moaks_shared_womac_drop_right_mbm_lasso_df.columns.duplicated()], v03_womac_drop_right_womac_columns)
v01_v03_moaks_womac_right_lasso_model, v01_v03_moaks_womac_right_lasso_metrics, v01_v03_moaks_womac_right_lasso_coefs_df = run_xgb_multioutput_regressor(v01_v03_moaks_shared_womac_drop_right_mbm_lasso_df.loc[:, ~v01_v03_moaks_shared_womac_drop_right_mbm_lasso_df.columns.duplicated()], v03_womac_drop_right_womac_columns)
