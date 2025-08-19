import pandas as pd
from config import moaks_shared_womac
from missing_imputation import moaks_shared_womac_drop_df, moaks_shared_womac_median_df
from modules.classes.oai_dataframe import OAI_DataFrame
from modules.variable_analysis import align_to_index

v01_moaks_shared_womac_wm_df = moaks_shared_womac_drop_df[['ID','READPRJ','SIDE','V01WOMKPL','V01WOMKPR','V01WOMADLL','V01WOMADLR','V01WOMSTFL','V01WOMSTFR']]
v03_moaks_shared_womac_wm_df = moaks_shared_womac_drop_df[['ID','READPRJ','SIDE','V03WOMKPL','V03WOMKPR','V03WOMADLL','V03WOMADLR','V03WOMSTFL','V03WOMSTFR']]

ΔWOMADLL = v03_moaks_shared_womac_wm_df['V03WOMADLL'] - v01_moaks_shared_womac_wm_df['V01WOMADLL']
ΔWOMADLR = v03_moaks_shared_womac_wm_df['V03WOMADLR'] - v01_moaks_shared_womac_wm_df['V01WOMADLR']

ΔWOMKPL = v03_moaks_shared_womac_wm_df['V03WOMKPL'] - v01_moaks_shared_womac_wm_df['V01WOMKPL']
ΔWOMKPR = v03_moaks_shared_womac_wm_df['V03WOMKPR'] - v01_moaks_shared_womac_wm_df['V01WOMKPR']

ΔWOMSTFL = v03_moaks_shared_womac_wm_df['V03WOMSTFL'] - v01_moaks_shared_womac_wm_df['V01WOMSTFL']
ΔWOMSTFR = v03_moaks_shared_womac_wm_df['V03WOMSTFL'] - v01_moaks_shared_womac_wm_df['V01WOMSTFL']

womac_left_drop_disability_mcid = ΔWOMADLL.std() * 0.5
womac_right_drop_disability_mcid = ΔWOMADLR.std() * 0.5
womac_left_drop_pain_mcid = ΔWOMKPL.std() *0.5
womac_right_drop_pain_mcid = ΔWOMKPR.std() * 0.5
womac_left_drop_stiffness_mcid = ΔWOMSTFL.std() * 0.5
womac_left_drop_stiffness_mcid = ΔWOMSTFR.std() * 0.5

WOMADLLPRG = ΔWOMADLL.apply(lambda score: 'Improved' if score <= -womac_left_drop_disability_mcid else 'Progressor' if score >= womac_left_drop_disability_mcid else 'Non-Progressor')
WOMADLRPRG = ΔWOMADLR.apply(lambda score: 'Improved' if score <= -womac_right_drop_disability_mcid else 'Progressor' if score >= womac_right_drop_disability_mcid else 'Non-Progressor')

WOMKPLPRG = ΔWOMKPL.apply(lambda score: 'Improved' if score <= -womac_left_drop_pain_mcid else 'Progressor' if score >= womac_left_drop_pain_mcid else 'Non-Progressor')
WOMKPRPRG = ΔWOMKPR.apply(lambda score: 'Improved' if score <= -womac_right_drop_pain_mcid else 'Progressor' if score >= womac_right_drop_pain_mcid else 'Non-Progressor')

WOMSTFLPRG = ΔWOMKPL.apply(lambda score: 'Improved' if score <= -womac_left_drop_stiffness_mcid else 'Progressor' if score >= womac_left_drop_stiffness_mcid else 'Non-Progressor')
WOMSTFRPRG = ΔWOMKPL.apply(lambda score: 'Improved' if score <= -womac_left_drop_stiffness_mcid else 'Progressor' if score >= womac_left_drop_stiffness_mcid else 'Non-Progressor')

index = moaks_shared_womac[['ID','READPRJ','SIDE']].sort_values(by=['ID','READPRJ','SIDE']).reset_index(drop=True)

wm_change_womac_df = pd.DataFrame({
    'ΔWOMKPR': ΔWOMKPR,
    'ΔWOMKPL': ΔWOMKPL,
    'ΔWOMADLL': ΔWOMADLL,
    'ΔWOMADLR': ΔWOMADLR,
    'ΔWOMSTFL': ΔWOMSTFL,
    'ΔWOMSTFR': ΔWOMSTFR,
    'WOMKPLPRG': WOMKPLPRG,
    'WOMKPRPRG': WOMKPRPRG,
    'WOMADLLPRG': WOMADLLPRG,
    'WOMADLRPRG': WOMADLRPRG,
    'WOMSTFLPRG': WOMSTFLPRG,
    'WOMSTFRPRG': WOMSTFRPRG,
})

wm_change_womac_df = align_to_index(index, wm_change_womac_df)

# -- merge by ID, READPRJ, and SIDE -- #
wm_change_womac_df = moaks_shared_womac_drop_df[['ID', 'READPRJ', 'SIDE']].merge(
    wm_change_womac_df,
    left_index=True,
    right_index=True,
    how='left'
)

wm_change_womac = OAI_DataFrame(wm_change_womac_df)

wm_change_womac_df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/wm_change_womac.csv", index=False)