from config import shared_knees_kl, v00_moaks_shared_kl

# -- baseline FMA columns in the FM (3757 knees) -- #
v00_moaks_shared_kl_fma_df = v00_moaks_shared_kl[['V00MBMSFMA', 'V00MBMNFMA', 'V00MBMPFMA'], shared_knees_kl]
# -- baseline FMC columns in the FM (3757 knees) -- #
v00_moaks_shared_kl_fmc_df = v00_moaks_shared_kl[['V00MBMSFMC', 'V00MBMNFMC', 'V00MBMPFMC'], shared_knees_kl]
# -- baseline FMP columns in the FM (3757 knees) -- #
v00_moaks_shared_kl_fmp_df = v00_moaks_shared_kl[['V00MBMSFMP', 'V00MBMNFMP', 'V00MBMPFMP'], shared_knees_kl]

# -- baseline MBMP load in the FM (3757 knees) -- #
V00FMAMBMLOAD = v00_moaks_shared_kl_fma_df.sum(axis=1)
# -- baseline MBMP load in the FM (3757 knees) -- #
V00FMCMBMLOAD = v00_moaks_shared_kl_fmc_df.sum(axis=1)
# -- baseline MBMP load in the FM (3757 knees) -- #
V00FMPMBMLOAD = v00_moaks_shared_kl_fmp_df.sum(axis=1)

# -- adding load columns to shared baseline MOAKS dataset -- #
v00_moaks_shared_kl['V00FMAMBMLOAD'] = V00FMAMBMLOAD
v00_moaks_shared_kl['V00FMCMBMLOAD'] = V00FMCMBMLOAD
v00_moaks_shared_kl['V00FMPMBMLOAD'] = V00FMPMBMLOAD
