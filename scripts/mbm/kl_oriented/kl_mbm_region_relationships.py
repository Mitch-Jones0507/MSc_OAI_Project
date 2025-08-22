from missing_imputation import moaks_shared_kl_drop_df, moaks_shared_womac_df

v00_moaks_shared_kl_mbms_fm_df = moaks_shared_kl_drop_df[['V00MBMSFMA','V00MBMSFMC','V00MBMSFMP']]
v00_moaks_shared_kl_mbmn_fm_df = moaks_shared_kl_drop_df[['V00MBMNFMA','V00MBMNFMC','V00MBMNFMP']]
v00_moaks_shared_kl_mbmp_fm_df = moaks_shared_kl_drop_df[['V00MBMPFMA','V00MBMPFMC','V00MBMPFMP']]

v00_moaks_shared_kl_mbms_p_df = moaks_shared_kl_drop_df[['V00MBMSPM','V00MBMSPL']]
v00_moaks_shared_kl_mbmn_p_df = moaks_shared_kl_drop_df[['V00MBMNPM','V00MBMNPL']]
v00_moaks_shared_kl_mbmp_p_df = moaks_shared_kl_drop_df[['V00MBMPPM','V00MBMPPL']]
