import pandas as pd
from config import moaks_shared_womac
from modules.classes.oai_dataframe import OAI_DataFrame
from modules.variable_analysis import factor_analysis, pca, align_to_index

# ========== Constructing mbm dataframes and dataframe objects ========== #

v00_moaks_shared_womac_mbms_df = moaks_shared_womac[['ID','READPRJ','SIDE','V00MBMSFMA','V00MBMSFMC','V00MBMSFMP',
'V00MBMSTMA','V00MBMSTMC','V00MBMSTMP']]

v00_moaks_shared_womac_mbmn_df = moaks_shared_womac[['ID','READPRJ','SIDE','V00MBMNFMA','V00MBMNFMC','V00MBMNFMP',
'V00MBMNTMA','V00MBMNTMC','V00MBMNTMP']]

v00_moaks_shared_womac_mbmp_df = moaks_shared_womac[['ID','READPRJ','SIDE','V00MBMPFMA','V00MBMPFMC','V00MBMPFMP',
'V00MBMPTMA','V00MBMPTMC','V00MBMPTMP']]

v01_moaks_shared_womac_mbms_df = moaks_shared_womac[['ID','READPRJ','SIDE','V01MBMSFMA','V01MBMSFMC','V01MBMSFMP',
'V01MBMSTMA','V01MBMSTMC','V01MBMSTMP']]

v01_moaks_shared_womac_mbmn_df = moaks_shared_womac[['ID','READPRJ','SIDE','V01MBMNFMA','V01MBMNFMC','V01MBMNFMP',
'V01MBMNTMA','V01MBMNTMC','V01MBMNTMP']]

v01_moaks_shared_womac_mbmp_df = moaks_shared_womac[['ID','READPRJ','SIDE','V01MBMPFMA','V01MBMPFMC','V01MBMPFMP',
'V01MBMPTMA','V01MBMPTMC','V01MBMPTMP']]

v00_moaks_shared_womac_mbms = OAI_DataFrame(v00_moaks_shared_womac_mbms_df)
v00_moaks_shared_womac_mbmn = OAI_DataFrame(v00_moaks_shared_womac_mbmn_df)
v00_moaks_shared_womac_mbmp = OAI_DataFrame(v00_moaks_shared_womac_mbmp_df)

v01_moaks_shared_womac_mbms = OAI_DataFrame(v01_moaks_shared_womac_mbms_df)
v01_moaks_shared_womac_mbmn = OAI_DataFrame(v01_moaks_shared_womac_mbmn_df)
v01_moaks_shared_womac_mbmp = OAI_DataFrame(v01_moaks_shared_womac_mbmp_df)

# ========== Computing MBM load (fm and tfm) scores using sum ========== #

# -- compute v00_mbm_fm_loads using simple sum -- #
V00MBMSFMLOADSUM = v00_moaks_shared_womac_mbms_df[['V00MBMSFMA','V00MBMSFMC','V00MBMSFMP']].sum(axis=1)
V00MBMNFMLOADSUM = v00_moaks_shared_womac_mbmn_df[['V00MBMNFMA','V00MBMNFMC','V00MBMNFMP']].sum(axis=1)
V00MBMPFMLOADSUM = v00_moaks_shared_womac_mbmp_df[['V00MBMPFMA','V00MBMPFMC','V00MBMPFMP']].sum(axis=1)

# -- compute v00_mbm_tfm_loads using simple sum -- #
V00MBMSTFMLOADSUM = v00_moaks_shared_womac_mbms_df[['V00MBMSFMA','V00MBMSFMC','V00MBMSFMP','V00MBMSTMA','V00MBMSTMC','V00MBMSTMP']].sum(axis=1)
V00MBMNTFMLOADSUM = v00_moaks_shared_womac_mbmn_df[['V00MBMNFMA','V00MBMNFMC','V00MBMNFMP','V00MBMNTMA','V00MBMNTMC','V00MBMNTMP']].sum(axis=1)
V00MBMPTFMLOADSUM = v00_moaks_shared_womac_mbmp_df[['V00MBMPFMA','V00MBMPFMC','V00MBMPFMP','V00MBMPTMA','V00MBMPTMC','V00MBMPTMP']].sum(axis=1)

# -- compute v01_mbm_fm_loads using simple sum -- #
V01MBMSFMLOADSUM = v01_moaks_shared_womac_mbms_df[['V01MBMSFMA','V01MBMSFMC','V01MBMSFMP']].sum(axis=1)
V01MBMNFMLOADSUM = v01_moaks_shared_womac_mbmn_df[['V01MBMNFMA','V01MBMNFMC','V01MBMNFMP']].sum(axis=1)
V01MBMPFMLOADSUM = v01_moaks_shared_womac_mbmp_df[['V01MBMPFMA','V01MBMPFMC','V01MBMPFMP']].sum(axis=1)

# -- compute v01_mbm_tfm_loads using simple sum -- #
V01MBMSTFMLOADSUM = v01_moaks_shared_womac_mbms_df[['V01MBMSFMA','V01MBMSFMC','V01MBMSFMP','V01MBMSTMA','V01MBMSTMC','V01MBMSTMP']].sum(axis=1)
V01MBMNTFMLOADSUM = v01_moaks_shared_womac_mbmn_df[['V01MBMNFMA','V01MBMNFMC','V01MBMNFMP','V01MBMNTMA','V01MBMNTMC','V01MBMNTMP']].sum(axis=1)
V01MBMPTFMLOADSUM = v01_moaks_shared_womac_mbmp_df[['V01MBMPFMA','V01MBMPFMC','V01MBMPFMP','V01MBMPTMA','V01MBMPTMC','V01MBMPTMP']].sum(axis=1)

# -- compute difference in mbm_fm_load sums between baseline and 12-months -- #
ΔMBMSFMLOADSUM = V01MBMSFMLOADSUM - V00MBMSFMLOADSUM
ΔMBMNFMLOADSUM = V01MBMNFMLOADSUM - V00MBMNFMLOADSUM
ΔMBMPFMLOADSUM = V01MBMPFMLOADSUM - V00MBMPFMLOADSUM

ΔMBMSTFMLOADSUM = V01MBMSTFMLOADSUM - V00MBMSTFMLOADSUM
ΔMBMNTFMLOADSUM = V01MBMNTFMLOADSUM - V00MBMNTFMLOADSUM
ΔMBMPTFMLOADSUM = V01MBMPTFMLOADSUM - V00MBMPTFMLOADSUM

# ========== Computing MBM load scores using factor analysis ========== #

# -- compute v00_mbm_fm_load factors -- #
V00MBMSFMLOADF, V00MBMSFMLOADF_loadings = factor_analysis(v00_moaks_shared_womac_mbms[['V00MBMSFMA','V00MBMSFMC','V00MBMSFMP']])
V00MBMNFMLOADF, V00MBMNFMLOADF_loadings = factor_analysis(v00_moaks_shared_womac_mbmn[['V00MBMNFMA','V00MBMNFMC','V00MBMNFMP']])
V00MBMPFMLOADF, V00MBMPFMLOADF_loadings = factor_analysis(v00_moaks_shared_womac_mbmp[['V00MBMPFMA','V00MBMPFMC','V00MBMPFMP']])

# -- compute v00_mbm_tfm_load factors -- #
V00MBMSTFMLOADF, V00MBMSTFMLOADF_loadings = factor_analysis(v00_moaks_shared_womac_mbms[:])
V00MBMNTFMLOADF, V00MBMNTFMLOADF_loadings = factor_analysis(v00_moaks_shared_womac_mbmn[:])
V00MBMPTFMLOADF, V00MBMPTFMLOADF_loadings = factor_analysis(v00_moaks_shared_womac_mbmp[:])

# -- compute v01_mbm_fm_load factors using loadings from v00_mbm factors -- #
V01MBMSFMLOADF, _ = factor_analysis(v01_moaks_shared_womac_mbms[['V01MBMSFMA','V01MBMSFMC','V01MBMSFMP']], V00MBMSFMLOADF_loadings)
V01MBMNFMLOADF, _ = factor_analysis(v01_moaks_shared_womac_mbmn[['V01MBMNFMA','V01MBMNFMC','V01MBMNFMP']], V00MBMNFMLOADF_loadings)
V01MBMPFMLOADF, _ = factor_analysis(v01_moaks_shared_womac_mbmp[['V01MBMPFMA','V01MBMPFMC','V01MBMPFMP']], V00MBMPFMLOADF_loadings)

V01MBMSTFMLOADF, _ = factor_analysis(v01_moaks_shared_womac_mbms[:], V00MBMSTFMLOADF_loadings)
V01MBMNTFMLOADF, _ = factor_analysis(v01_moaks_shared_womac_mbmn[:], V00MBMNTFMLOADF_loadings)
V01MBMPTFMLOADF, _ = factor_analysis(v01_moaks_shared_womac_mbmp[:], V00MBMPTFMLOADF_loadings)

# -- compute difference in mbm_fm_load factors between baseline and 12-months -- #
ΔMBMSFMLOADF = V01MBMSFMLOADF - V00MBMSFMLOADF
ΔMBMNFMLOADF = V01MBMNFMLOADF - V00MBMNFMLOADF
ΔMBMPFMLOADF = V01MBMPFMLOADF - V00MBMPFMLOADF

# -- compute difference in mbm_tfm_load factors between baseline and 12-months -- #
ΔMBMSTFMLOADF = V01MBMSTFMLOADF - V00MBMSTFMLOADF
ΔMBMNTFMLOADF = V01MBMNTFMLOADF - V00MBMNTFMLOADF
ΔMBMPTFMLOADF = V01MBMPTFMLOADF - V00MBMPTFMLOADF

# ========== Computing MBM load scores using principal component analysis ========== #

# -- compute v00_mbm_fm_load first principal components -- #
V00MBMSFMLOADP, V00MBMSFMLOADP_model, V00MBMSFMLOADP_statistics = pca(v00_moaks_shared_womac_mbms[['V00MBMSFMA','V00MBMSFMC','V00MBMSFMP']])
V00MBMNFMLOADP, V00MBMNFMLOADP_model, V00MBMNFMLOADP_statistics = pca(v00_moaks_shared_womac_mbmn[['V00MBMNFMA','V00MBMNFMC','V00MBMNFMP']])
V00MBMPFMLOADP, V00MBMPFMLOADP_model, V00MBMPFMLOADP_statistics = pca(v00_moaks_shared_womac_mbmp[['V00MBMPFMA','V00MBMPFMC','V00MBMPFMP']])

# -- compute v00_mbm_tfm_load first principal components -- #
V00MBMSTFMLOADP, V00MBMSTFMLOADP_model, V00MBMSTFMLOADP_statistics = pca(v00_moaks_shared_womac_mbms[:])
V00MBMNTFMLOADP, V00MBMNTFMLOADP_model, V00MBMNTFMLOADP_statistics = pca(v00_moaks_shared_womac_mbmn[:])
V00MBMPTFMLOADP, V00MBMPTFMLOADP_model, V00MBMPTFMLOADP_statistics = pca(v00_moaks_shared_womac_mbmp[:])

# -- compute v01_mbm_fm_load first principal components -- #
V01MBMSFMLOADP, _, _ = pca(v01_moaks_shared_womac_mbms[['V01MBMSFMA','V01MBMSFMC','V01MBMSFMP']], V00MBMSFMLOADP_model, V00MBMSFMLOADP_statistics)
V01MBMNFMLOADP, _, _ = pca(v01_moaks_shared_womac_mbmn[['V01MBMNFMA','V01MBMNFMC','V01MBMNFMP']], V00MBMNFMLOADP_model, V00MBMNFMLOADP_statistics)
V01MBMPFMLOADP, _, _ = pca(v01_moaks_shared_womac_mbmp[['V01MBMPFMA','V01MBMPFMC','V01MBMPFMP']], V00MBMPFMLOADP_model, V00MBMPFMLOADP_statistics)

# -- compute v01_mbm_tfm_load first principal components -- #
V01MBMSTFMLOADP, _, _ = pca(v01_moaks_shared_womac_mbms[:], V00MBMSTFMLOADP_model, V00MBMSTFMLOADP_statistics)
V01MBMNTFMLOADP, _, _ = pca(v01_moaks_shared_womac_mbmn[:], V00MBMNTFMLOADP_model, V00MBMNTFMLOADP_statistics)
V01MBMPTFMLOADP, _, _ = pca(v01_moaks_shared_womac_mbmp[:], V00MBMPTFMLOADP_model, V00MBMPTFMLOADP_statistics)

# -- compute difference in mbm_fm_load first principal components -- #
ΔMBMSFMLOADP = V01MBMSFMLOADP - V00MBMSFMLOADP
ΔMBMNFMLOADP = V01MBMNFMLOADP - V00MBMNFMLOADP
ΔMBMPFMLOADP = V01MBMPFMLOADP - V00MBMPFMLOADP

# -- compute difference in mbm_fm_load first principal components -- #
ΔMBMSTFMLOADP = V01MBMSTFMLOADP - V00MBMSTFMLOADP
ΔMBMNTFMLOADP = V01MBMNTFMLOADP - V00MBMNTFMLOADP
ΔMBMPTFMLOADP = V01MBMPTFMLOADP - V00MBMPTFMLOADP

# ========== Building MBM load dataframe object ========== #

# -- esnure all columns are aligned by ID, READPRJ, and SIDE -- #
index = moaks_shared_womac[['ID','READPRJ','SIDE']].sort_values(by=['ID','READPRJ','SIDE']).reset_index(drop=True)

# -- construct mbm load dataframe with mbm load columns -- #
mbm_load_womac_df = pd.DataFrame({
    'V00MBMSFMLOADSUM': V00MBMSFMLOADSUM,
    'V00MBMNFMLOADSUM': V00MBMNFMLOADSUM,
    'V00MBMPFMLOADSUM': V00MBMPFMLOADSUM,
    'V00MBMSTFMLOADSUM': V00MBMSTFMLOADSUM,
    'V00MBMNTFMLOADSUM': V00MBMNTFMLOADSUM,
    'V00MBMPTFMLOADSUM': V00MBMPTFMLOADSUM,
    'V01MBMSFMLOADSUM': V01MBMSFMLOADSUM,
    'V01MBMNFMLOADSUM': V01MBMNFMLOADSUM,
    'V01MBMPFMLOADSUM': V01MBMPFMLOADSUM,
    'V01MBMSTFMLOADSUM': V01MBMSTFMLOADSUM,
    'V01MBMNTFMLOADSUM': V01MBMNTFMLOADSUM,
    'V01MBMPTFMLOADSUM': V01MBMPTFMLOADSUM,
    'ΔMBMSFMLOADSUM': ΔMBMSFMLOADSUM,
    'ΔMBMNFMLOADSUM': ΔMBMNFMLOADSUM,
    'ΔMBMPFMLOADSUM': ΔMBMPFMLOADSUM,
    'ΔMBMSTFMLOADSUM': ΔMBMSTFMLOADSUM,
    'ΔMBMNTFMLOADSUM': ΔMBMNTFMLOADSUM,
    'ΔMBMPTFMLOADSUM': ΔMBMPTFMLOADSUM,
    'V00MBMSFMLOADF': V00MBMSFMLOADF,
    'V00MBMNFMLOADF': V00MBMNFMLOADF,
    'V00MBMPFMLOADF': V00MBMPFMLOADF,
    'V00MBMSTFMLOADF': V00MBMSTFMLOADF,
    'V00MBMNTFMLOADF': V00MBMNTFMLOADF,
    'V00MBMPTFMLOADF': V00MBMPTFMLOADF,
    'V01MBMSFMLOADF': V01MBMSFMLOADF,
    'V01MBMNFMLOADF': V01MBMNFMLOADF,
    'V01MBMPFMLOADF': V01MBMPFMLOADF,
    'V01MBMSTFMLOADF': V01MBMSTFMLOADF,
    'V01MBMNTFMLOADF': V01MBMNTFMLOADF,
    'V01MBMPTFMLOADF': V01MBMPTFMLOADF,
    'ΔMBMSFMLOADF': ΔMBMSFMLOADF,
    'ΔMBMNFMLOADF': ΔMBMNFMLOADF,
    'ΔMBMPFMLOADF': ΔMBMPFMLOADF,
    'ΔMBMSTFMLOADF': ΔMBMSTFMLOADF,
    'ΔMBMNTFMLOADF': ΔMBMNTFMLOADF,
    'ΔMBMPTFMLOADF': ΔMBMPTFMLOADF,
    'V00MBMSFMLOADP': V00MBMSFMLOADP,
    'V00MBMNFMLOADP': V00MBMNFMLOADP,
    'V00MBMPFMLOADP': V00MBMPFMLOADP,
    'V00MBMSTFMLOADP': V00MBMSTFMLOADP,
    'V00MBMNTFMLOADP': V00MBMNTFMLOADP,
    'V00MBMPTFMLOADP': V00MBMPTFMLOADP,
    'V01MBMSFMLOADP': V01MBMSFMLOADP,
    'V01MBMNFMLOADP': V01MBMNFMLOADP,
    'V01MBMPFMLOADP': V01MBMPFMLOADP,
    'V01MBMSTFMLOADP': V01MBMSTFMLOADP,
    'V01MBMNTFMLOADP': V01MBMNTFMLOADP,
    'V01MBMPTFMLOADP': V01MBMPTFMLOADP,
    'ΔMBMSFMLOADP': ΔMBMPFMLOADP,
    'ΔMBMNFMLOADP': ΔMBMNFMLOADP,
    'ΔMBMPFMLOADP': ΔMBMPFMLOADP,
    'ΔMBMSTFMLOADP': ΔMBMPTFMLOADP,
    'ΔMBMNTFMLOADP': ΔMBMNTFMLOADP,
    'ΔMBMPTFMLOADP': ΔMBMPTFMLOADP
})

# -- align mbm load dataframe -- #
mbm_load_womac_df = align_to_index(index, mbm_load_womac_df)

# -- merge by ID, READPRJ, and SIDE -- #
mbm_load_womac_df = moaks_shared_womac[['ID', 'READPRJ', 'SIDE']].merge(
    mbm_load_womac_df,
    left_index=True,
    right_index=True,
    how='left'
)
# -- mbm load dataset object -- #
mbm_load_womac = OAI_DataFrame(mbm_load_womac_df)

# -- save to csv -- #
#mbm_load_womac_df.to_csv('/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/mbm_load_womac.csv', index=False)
