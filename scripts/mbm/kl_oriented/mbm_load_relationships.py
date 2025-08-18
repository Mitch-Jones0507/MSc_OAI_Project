from scripts.mbm.mbm_load import v00_moaks_shared_kl
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from modules.p_matrices import build_p_matrix, sort_p_matrix
from modules.variable_analysis import full_regression_analysis

# -- shared baseline mbm load dataframe -- #
v00_moaks_shared_kl_mbm_load_df = v00_moaks_shared_kl[['V00MBMSFMLOAD', 'V00MBMNFMLOAD', 'V00MBMPFMLOAD']]

# -- shared baseline mbm load object -- #
v00_moaks_shared_kl_mbm_load = MOAKS_DataFrame(v00_moaks_shared_kl_mbm_load_df)

mbm_relationships_df = full_regression_analysis(v00_moaks_shared_kl_mbm_load)

# -- build and sort p_value dataframe for each mbm target and predictor -- #
mbm_p_matrix_df = sort_p_matrix(build_p_matrix(mbm_relationships_df))

# -- adding to csv -- #
#mbm_relationships_df.to_csv('/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/mbm_relationships.csv', index=False)

# -- adding to csv -- #
#p_matrix_df.to_csv('/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/p_matrix.csv', index=False)