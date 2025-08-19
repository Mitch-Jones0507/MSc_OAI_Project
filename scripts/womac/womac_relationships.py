import numpy as np
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from modules.p_matrices import sort_p_matrix, build_p_matrix
from modules.variable_analysis import pca, full_regression_analysis
from scripts.womac.womac_distribution import moaks_shared_womac_drop_df, moaks_shared_womac_median_df
from scripts.womac.womac_change import wm_change_womac_df

v01_womac_left_drop_df = moaks_shared_womac_drop_df[['V01WOMADLL','V01WOMKPL','V01WOMSTFL']]
v03_womac_left_drop_df = moaks_shared_womac_drop_df[['V03WOMADLL', 'V03WOMKPL', 'V03WOMSTFL']]
womac_left_drop_change_df = wm_change_womac_df[['ΔWOMADLL','ΔWOMKPL','ΔWOMSTFL']]

v01_womac_left_median_df = moaks_shared_womac_median_df[['V01WOMADLL', 'V01WOMKPL', 'V01WOMSTFL']]
v03_womac_left_median_df = moaks_shared_womac_median_df[['V03WOMADLL', 'V03WOMKPL', 'V03WOMSTFL']]

v01_womac_right_drop_df = moaks_shared_womac_drop_df[['V01WOMADLR','V01WOMKPR','V01WOMSTFR']]
v03_womac_right_drop_df = moaks_shared_womac_drop_df[['V03WOMADLR', 'V03WOMKPR', 'V03WOMSTFR']]
womac_right_drop_change_df = wm_change_womac_df[['ΔWOMADLR','ΔWOMKPR','ΔWOMSTFR']]

v01_womac_right_median_df = moaks_shared_womac_median_df[['V01WOMADLR', 'V01WOMKPR', 'V01WOMSTFR']]
v03_womac_right_median_df = moaks_shared_womac_median_df[['V03WOMADLR', 'V03WOMKPR', 'V03WOMSTFR']]

v01_womac_left_drop_reduced_df, v01_womac_left_drop_model, _ = pca(v01_womac_left_drop_df,components = 3)

v01_womac_left_drop_pc_variance = v01_womac_left_drop_model.explained_variance_
v01_womac_left_drop_pc_variance_ratio = v01_womac_left_drop_model.explained_variance_ratio_
v01_womac_left_drop_pc_loadings = v01_womac_left_drop_model.components_.T * np.sqrt(v01_womac_left_drop_model.explained_variance_)

v03_womac_left_drop_reduced_df, v03_womac_left_drop_model, _ = pca(v03_womac_left_drop_df, components = 3)

v03_womac_left_drop_pc_variance = v03_womac_left_drop_model.explained_variance_
v03_womac_left_drop_pc_variance_ratio = v03_womac_left_drop_model.explained_variance_ratio_
v03_womac_left_drop_pc_loadings = v03_womac_left_drop_model.components_.T * np.sqrt(v03_womac_left_drop_model.explained_variance_)

womac_left_drop_change_reduced_df, womac_left_drop_change_model, _ = pca(womac_left_drop_change_df,components = 3)

womac_left_drop_change_pc_variance = womac_left_drop_change_model.explained_variance_
womac_left_drop_change_pc_variance_ratio = womac_left_drop_change_model.explained_variance_ratio_
womac_left_drop_change_pc_loadings = womac_left_drop_change_model.components_.T * np.sqrt(womac_left_drop_change_model.explained_variance_)

v01_womac_left_median_reduced_df, v01_womac_left_median_model, _ = pca(v01_womac_left_median_df, components = 3)

v01_womac_left_median_pc_variance = v01_womac_left_median_model.explained_variance_
v01_womac_left_median_pc_variance_ratio = v01_womac_left_median_model.explained_variance_ratio_
v01_womac_left_median_pc_loadings = v01_womac_left_median_model.components_.T * np.sqrt(v01_womac_left_median_model.explained_variance_)

v03_womac_left_median_reduced_df, v03_womac_left_median_model, _ = pca(v03_womac_left_median_df, components = 3)

v03_womac_left_median_pc_variance = v03_womac_left_median_model.explained_variance_
v03_womac_left_median_pc_variance_ratio = v03_womac_left_median_model.explained_variance_ratio_
v03_womac_left_median_pc_loadings = v03_womac_left_median_model.components_.T * np.sqrt(v03_womac_left_median_model.explained_variance_)

v01_womac_right_drop_reduced_df, v01_womac_right_drop_model, _ = pca(v01_womac_right_drop_df,components = 3)

v01_womac_right_drop_pc_variance = v01_womac_right_drop_model.explained_variance_
v01_womac_right_drop_pc_variance_ratio = v01_womac_right_drop_model.explained_variance_ratio_
v01_womac_right_drop_pc_loadings = v01_womac_right_drop_model.components_.T * np.sqrt(v01_womac_right_drop_model.explained_variance_)

v03_womac_right_drop_reduced_df, v03_womac_right_drop_model, _ = pca(v03_womac_right_drop_df, components = 3)

v03_womac_right_drop_pc_variance = v03_womac_right_drop_model.explained_variance_
v03_womac_right_drop_pc_variance_ratio = v03_womac_right_drop_model.explained_variance_ratio_
v03_womac_right_drop_pc_loadings = v03_womac_right_drop_model.components_.T * np.sqrt(v03_womac_right_drop_model.explained_variance_)

womac_right_drop_change_reduced_df, womac_right_drop_change_model, _ = pca(womac_right_drop_change_df,components = 3)

womac_right_drop_change_pc_variance = womac_right_drop_change_model.explained_variance_
womac_right_drop_change_pc_variance_ratio = womac_right_drop_change_model.explained_variance_ratio_
womac_right_drop_change_pc_loadings = womac_right_drop_change_model.components_.T * np.sqrt(womac_right_drop_change_model.explained_variance_)

v01_womac_right_median_reduced_df, v01_womac_right_median_model, _ = pca(v01_womac_right_median_df, components = 3)

v01_womac_right_median_pc_variance = v01_womac_right_median_model.explained_variance_
v01_womac_right_median_pc_variance_ratio = v01_womac_right_median_model.explained_variance_ratio_
v01_womac_right_median_pc_loadings = v01_womac_right_median_model.components_.T * np.sqrt(v01_womac_right_median_model.explained_variance_)

v03_womac_right_median_reduced_df, v03_womac_right_median_model, _ = pca(v03_womac_right_median_df, components = 3)

v03_womac_right_median_pc_variance = v03_womac_right_median_model.explained_variance_
v03_womac_right_median_pc_variance_ratio = v03_womac_right_median_model.explained_variance_ratio_
v03_womac_right_median_pc_loadings = v03_womac_right_median_model.components_.T * np.sqrt(v03_womac_right_median_model.explained_variance_)

v01_womac_left_drop = MOAKS_DataFrame(v01_womac_left_drop_df)
v03_womac_left_drop = MOAKS_DataFrame(v03_womac_left_drop_df)
womac_left_drop_change = MOAKS_DataFrame(womac_left_drop_change_df)
v01_womac_right_drop = MOAKS_DataFrame(v01_womac_right_drop_df)
v03_womac_right_drop = MOAKS_DataFrame(v03_womac_right_drop_df)
womac_right_drop_change = MOAKS_DataFrame(womac_right_drop_change_df)

v01_womac_left_median = MOAKS_DataFrame(v01_womac_left_median_df)
v03_womac_left_median = MOAKS_DataFrame(v03_womac_left_median_df)
v01_womac_right_median = MOAKS_DataFrame(v01_womac_right_median_df)
v03_womac_right_median = MOAKS_DataFrame(v03_womac_right_median_df)

v01_womac_left_drop_regression_summary = full_regression_analysis(v01_womac_left_drop)
v03_womac_left_drop_regression_summary = full_regression_analysis(v03_womac_left_drop)
womac_left_drop_change_regression_summary = full_regression_analysis(womac_left_drop_change)
v01_womac_right_drop_regression_summary = full_regression_analysis(v01_womac_right_drop)
v03_womac_right_drop_regression_summary = full_regression_analysis(v03_womac_right_drop)
womac_right_drop_change_regression_summary = full_regression_analysis(womac_right_drop_change)

v01_womac_left_drop_p_matrix = sort_p_matrix(build_p_matrix(v01_womac_left_drop_regression_summary))
v03_womac_left_drop_p_matrix = sort_p_matrix(build_p_matrix(v03_womac_left_drop_regression_summary))
womac_left_drop_change_p_matrix = sort_p_matrix(build_p_matrix(womac_left_drop_change_regression_summary))
v01_womac_right_drop_p_matrix = sort_p_matrix(build_p_matrix(v01_womac_right_drop_regression_summary))
v03_womac_right_drop_p_matrix = sort_p_matrix(build_p_matrix(v03_womac_right_drop_regression_summary))
womac_right_drop_change_p_matrix = sort_p_matrix(build_p_matrix(womac_right_drop_change_regression_summary))

v01_womac_left_median_regression_summary = full_regression_analysis(v01_womac_left_median)
v03_womac_left_median_regression_summary = full_regression_analysis(v03_womac_left_median)
v01_womac_right_median_regression_summary = full_regression_analysis(v01_womac_right_median)
v03_womac_right_median_regression_summary = full_regression_analysis(v03_womac_right_median)

v01_womac_left_median_p_matrix = sort_p_matrix(build_p_matrix(v01_womac_left_median_regression_summary))
v03_womac_left_median_p_matrix = sort_p_matrix(build_p_matrix(v03_womac_left_median_regression_summary))
v01_womac_right_median_p_matrix = sort_p_matrix(build_p_matrix(v01_womac_right_median_regression_summary))
v03_womac_right_median_p_matrix = sort_p_matrix(build_p_matrix(v03_womac_right_median_regression_summary))

# ========= Construct reduced target vectors based on PCA/regression analysis ========== #

moaks_shared_womac_drop_df[['V01LPC1','V01LPC2']] =  v01_womac_left_drop_reduced_df[['PC1','PC2']]
moaks_shared_womac_drop_df[['V03LPC1','V03LPC2']] = v03_womac_left_drop_reduced_df[['PC1','PC2']]
moaks_shared_womac_drop_df[['ΔLPC1','ΔLPC2']] = womac_left_drop_change_reduced_df[['PC1','PC2']]

moaks_shared_womac_drop_df[['V01RPC1','V01RPC2']] = v01_womac_right_drop_reduced_df[['PC1','PC2']]
moaks_shared_womac_drop_df[['V03RPC1','V03RPC2']] = v03_womac_right_drop_reduced_df[['PC1','PC2']]
moaks_shared_womac_drop_df[['ΔRPC1','ΔRPC2']] = womac_right_drop_change_reduced_df[['PC1','PC2']]
