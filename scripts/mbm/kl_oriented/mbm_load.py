from config import v00_v01_shared_knees_kl, v00_moaks_shared_kl
from modules.variable_analysis import factor_analysis

# -- baseline MBMS columns in the FM (3077 knees) -- #
v00_moaks_shared_kl_mbms_df = v00_moaks_shared_kl[['V00MBMSFMA', 'V00MBMSFMC', 'V00MBMSFMP'], v00_v01_shared_knees_kl]
# -- baseline MBMN columns in the FM (3077 knees) -- #
v00_moaks_shared_kl_mbmn_df = v00_moaks_shared_kl[['V00MBMNFMA', 'V00MBMNFMC', 'V00MBMNFMP'], v00_v01_shared_knees_kl]
# -- baseline MBMP columns in the FM (3077 knees) -- #
v00_moaks_shared_kl_mbmp_df = v00_moaks_shared_kl[['V00MBMPFMA', 'V00MBMPFMC', 'V00MBMPFMP'], v00_v01_shared_knees_kl]

# -- baseline MBMP load in the FM (3077 knees) -- #
V00MBMSFMLOAD = v00_moaks_shared_kl_mbms_df.sum(axis=1)
# -- baseline MBMP load in the FM (3077 knees) -- #
V00MBMNFMLOAD = v00_moaks_shared_kl_mbmn_df.sum(axis=1)
# -- baseline MBMP load in the FM (3077 knees) -- #
V00MBMPFMLOAD = v00_moaks_shared_kl_mbmp_df.sum(axis=1)

# -- adding load columns to shared baseline MOAKS dataset -- #
v00_moaks_shared_kl['V00MBMSFMLOAD'] = V00MBMSFMLOAD
v00_moaks_shared_kl['V00MBMNFMLOAD'] = V00MBMNFMLOAD
v00_moaks_shared_kl['V00MBMPFMLOAD'] = V00MBMPFMLOAD

# -- adding to csv -- #
#v00_moaks_shared.df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/v00_moaks_shared_mbm_load.csv",index=False)