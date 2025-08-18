from scripts.fm.fm_load import v00_moaks_shared_kl
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from modules.p_matrices import build_p_matrix, sort_p_matrix
from modules.variable_analysis import full_regression_analysis

v00_moaks_shared_kl_fm_load_df = v00_moaks_shared_kl[['V00FMAMBMLOAD','V00FMCMBMLOAD','V00FMPMBMLOAD']]

v00_moaks_shared_kl_fm_load = MOAKS_DataFrame(v00_moaks_shared_kl_fm_load_df)

fm_relationships_df = full_regression_analysis(v00_moaks_shared_kl_fm_load)

fm_p_matrix = sort_p_matrix(build_p_matrix(fm_relationships_df))

