import pandas as pd
from config import *

# -- baseline MBMS columns in the FM (3077 knees) -- #
v00_moaks_shared_mbms = v00_moaks_shared[['V00MBMSFMA','V00MBMSFMC','V00MBMSFMP'], shared_knees_kl]
# -- baseline MBMN columns in the FM (3077 knees) -- #
v00_moaks_shared_mbmn = v00_moaks_shared[['V00MBMNFMA','V00MBMNFMC','V00MBMNFMP'], shared_knees_kl]
# -- baseline MBMP columns in the FM (3077 knees) -- #
v00_moaks_shared_mbmp = v00_moaks_shared[['V00MBMPFMA','V00MBMPFMC','V00MBMPFMP'], shared_knees_kl]

# -- baseline MBMP load in the FM (3077 knees) -- #
V00MBMSFMLOAD = v00_moaks_shared_mbms.sum(axis=1)
# -- baseline MBMP load in the FM (3077 knees) -- #
V00MBMNFMLOAD = v00_moaks_shared_mbmn.sum(axis=1)
# -- baseline MBMP load in the FM (3077 knees) -- #
V00MBMPFMLOAD = v00_moaks_shared_mbmp.sum(axis=1)

# -- adding load columns to shared baseline MOAKS dataset -- #
v00_moaks_shared['V00MBMSFMLOAD'] = V00MBMSFMLOAD
v00_moaks_shared['V00MBMNFMLOAD'] = V00MBMNFMLOAD
v00_moaks_shared['V00MBMPFMLOAD'] = V00MBMPFMLOAD

# -- adding to csv -- #
#v00_moaks_shared.df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/v00_moaks_shared_mbm_load.csv",index=False)
