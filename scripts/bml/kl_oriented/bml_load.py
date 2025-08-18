import warnings
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from scripts.mbm.mbm_load_relationships import v00_moaks_shared_kl_mbm_load, v00_moaks_shared_kl_mbm_load_df
from modules.variable_analysis import factor_analysis, pca
warnings.simplefilter(action='ignore', category=FutureWarning)

v00_moaks_shared_kl_mbm_load_df = v00_moaks_shared_kl_mbm_load_df.copy()

bml_load_factors = factor_analysis(v00_moaks_shared_kl_mbm_load)[0]
v00_moaks_shared_kl_mbm_load_df['V00BMLFMLOADF'] = [load[0] for load in bml_load_factors]

bml_load_pca = pca(v00_moaks_shared_kl_mbm_load)
v00_moaks_shared_kl_mbm_load_df['V00BMLFMLOADP'] = bml_load_pca

bml_load_sum = v00_moaks_shared_kl_mbm_load[['V00MBMSFMLOAD', 'V00MBMNFMLOAD', 'V00MBMPFMLOAD']].sum(axis=1)
v00_moaks_shared_kl_mbm_load_df['V00BMLFMLOADSUM'] = bml_load_sum

bml_load_sum_no_mbmp = v00_moaks_shared_kl_mbm_load[['V00MBMSFMLOAD', 'V00MBMNFMLOAD']].sum(axis=1)
v00_moaks_shared_kl_mbm_load_df['V00BMLFMLOADSUMNOP'] = bml_load_sum_no_mbmp

v00_moaks_shared_kl_mbm_load_df.to_csv('/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/v00_moaks_shared_bml_load.csv', index=False)

v00_moaks_shared_kl_bml_load = MOAKS_DataFrame(v00_moaks_shared_kl_mbm_load_df)

print(v00_moaks_shared_kl_bml_load)