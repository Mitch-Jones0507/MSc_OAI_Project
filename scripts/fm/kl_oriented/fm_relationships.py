from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from modules.p_matrices import build_p_matrix, sort_p_matrix
from scripts.fm.fm_load import v00_moaks_shared_kl_fma_df, v00_moaks_shared_kl_fmc_df, v00_moaks_shared_kl_fmp_df
from modules.variable_analysis import full_regression_analysis

# -- shared baseline fma object -- #
v00_moaks_shared_kl_fma = MOAKS_DataFrame(v00_moaks_shared_kl_fma_df)
# -- shared baseline fmc object -- #
v00_moaks_shared_kl_fmc = MOAKS_DataFrame(v00_moaks_shared_kl_fmc_df)
# -- shared baseline fmp object -- #
v00_moaks_shared_kl_fmp = MOAKS_DataFrame(v00_moaks_shared_kl_fmp_df)

fma_relationships_df = full_regression_analysis(v00_moaks_shared_kl_fma)
fmc_relationships_df = full_regression_analysis(v00_moaks_shared_kl_fmc)
fmp_relationships_df = full_regression_analysis(v00_moaks_shared_kl_fmp)

fma_p_matrix = sort_p_matrix(build_p_matrix(fma_relationships_df))
fmc_p_matrix = sort_p_matrix(build_p_matrix(fmc_relationships_df))
fmp_p_matrix = sort_p_matrix(build_p_matrix(fmp_relationships_df))
