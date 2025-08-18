import numpy as np
from missing_imputation import moaks_shared_kl_drop_df, moaks_shared_kl_median_df
from modules.variable_analysis import pca

v00_moaks_shared_kl_mbm_columns = [column for column in moaks_shared_kl_drop_df.columns if 'V00MBM' in column]
v01_moaks_shared_kl_mbm_columns = [column for column in moaks_shared_kl_drop_df.columns if 'V01MBM' in column]

v00_moaks_shared_kl_drop_mbm_df = moaks_shared_kl_drop_df[v00_moaks_shared_kl_mbm_columns]
v01_moaks_shared_kl_drop_mbm_df = moaks_shared_kl_drop_df[v01_moaks_shared_kl_mbm_columns]
v00_moaks_shared_kl_median_mbm_df = moaks_shared_kl_median_df[v00_moaks_shared_kl_mbm_columns]
v01_moaks_shared_kl_median_mbm_df = moaks_shared_kl_median_df[v01_moaks_shared_kl_mbm_columns]

# ========== Compute PCA on drop dataset ========== #

v00_moaks_shared_kl_drop_mbm_reduced_df, v00_moaks_shared_kl_drop_mbm_model, _ = pca(v00_moaks_shared_kl_drop_mbm_df,components = 10)

v00_moaks_shared_kl_drop_mbm_pc_variance = v00_moaks_shared_kl_drop_mbm_model.explained_variance_
v00_moaks_shared_kl_drop_mbm_pc_variance_ratio = v00_moaks_shared_kl_drop_mbm_model.explained_variance_ratio_
v00_moaks_shared_kl_drop_mbm_pc_loadings = v00_moaks_shared_kl_drop_mbm_model.components_.T * np.sqrt(v00_moaks_shared_kl_drop_mbm_model.explained_variance_)

v01_moaks_shared_kl_drop_mbm_reduced_df, v01_moaks_shared_kl_drop_mbm_model, _ = pca(v01_moaks_shared_kl_drop_mbm_df,components = 10)

v01_moaks_shared_kl_drop_mbm_pc_variance = v01_moaks_shared_kl_drop_mbm_model.explained_variance_
v01_moaks_shared_kl_drop_mbm_pc_variance_ratio = v01_moaks_shared_kl_drop_mbm_model.explained_variance_ratio_
v01_moaks_shared_kl_drop_mbm_pc_loadings = v01_moaks_shared_kl_drop_mbm_model.components_.T * np.sqrt(v01_moaks_shared_kl_drop_mbm_model.explained_variance_)

# ========== Compute PCA on median dataset ========== #

v00_moaks_shared_kl_median_mbm_reduced_df, v00_moaks_shared_kl_median_mbm_model, _ = pca(v00_moaks_shared_kl_median_mbm_df,components = 10)

v00_moaks_shared_kl_median_mbm_pc_variance = v00_moaks_shared_kl_median_mbm_model.explained_variance_
v00_moaks_shared_kl_median_mbm_pc_variance_ratio = v00_moaks_shared_kl_median_mbm_model.explained_variance_ratio_
v00_moaks_shared_kl_median_mbm_pc_loadings = v00_moaks_shared_kl_median_mbm_model.components_.T * np.sqrt(v00_moaks_shared_kl_median_mbm_model.explained_variance_)

v01_moaks_shared_kl_median_mbm_reduced_df, v01_moaks_shared_kl_median_mbm_model, _ = pca(v01_moaks_shared_kl_median_mbm_df,components = 10)

v01_moaks_shared_kl_median_mbm_pc_variance = v01_moaks_shared_kl_median_mbm_model.explained_variance_
v01_moaks_shared_kl_median_mbm_pc_variance_ratio = v01_moaks_shared_kl_median_mbm_model.explained_variance_ratio_
v01_moaks_shared_kl_median_mbm_pc_loadings = v01_moaks_shared_kl_median_mbm_model.components_.T * np.sqrt(v01_moaks_shared_kl_median_mbm_model.explained_variance_)
