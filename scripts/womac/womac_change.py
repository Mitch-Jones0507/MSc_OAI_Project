import pandas as pd
from config import moaks_shared_womac
from modules.classes.oai_dataframe import OAI_DataFrame
from modules.plot_data import plot_predictor_target_data
from modules.variable_analysis import align_to_index

v01_moaks_shared_womac_wm_df = moaks_shared_womac[['ID','READPRJ','SIDE','V01WOMKPL','V01WOMKPR','V01WOMTSL','V01WOMTSR']]
v03_moaks_shared_womac_wm_df = moaks_shared_womac[['ID','READPRJ','SIDE','V03WOMKPL','V03WOMKPR','V03WOMTSL','V03WOMTSR']]

ΔWMKPR = v03_moaks_shared_womac_wm_df['V03WOMKPR'] - v01_moaks_shared_womac_wm_df['V01WOMKPR']
ΔWMKPL = v03_moaks_shared_womac_wm_df['V03WOMKPL'] - v01_moaks_shared_womac_wm_df['V01WOMKPL']

ΔWMTSR = v03_moaks_shared_womac_wm_df['V03WOMTSR'] - v01_moaks_shared_womac_wm_df['V01WOMTSR']
ΔWMTSL = v03_moaks_shared_womac_wm_df['V03WOMTSL'] - v01_moaks_shared_womac_wm_df['V01WOMTSL']
ΔWMTS = ΔWMTSR + ΔWMTSL

mcid_pain = 5
mcid_total = 10
mcid_total_both_knee = 20

WMPPRGR = ΔWMKPR.apply(lambda score: 'Improved' if score <= -mcid_pain else 'Progressor' if score >= mcid_pain else 'Non-Progressor')
WMPPRGL = ΔWMKPL.apply(lambda score: 'Improved' if score <= -mcid_pain else 'Progressor' if score >= mcid_pain else 'Non-Progressor')

WMPRGR = ΔWMTSR.apply(lambda score: 'Improved' if score <= -mcid_total else 'Progressor' if score >= mcid_total else 'Non-Progressor')
WMPRGL = ΔWMTSL.apply(lambda score: 'Improved' if score <= -mcid_total else 'Progressor' if score >= mcid_total else 'Non-Progressor')

WMPRG = ΔWMTS.apply(lambda score: 'Improved' if score <= -mcid_total_both_knee else 'Progressor' if score >= mcid_total_both_knee else 'Non-Progressor')

index = moaks_shared_womac[['ID','READPRJ','SIDE']].sort_values(by=['ID','READPRJ','SIDE']).reset_index(drop=True)

wm_change_womac_df = pd.DataFrame({
    'ΔWMKPR': ΔWMKPR,
    'ΔWMKPL': ΔWMKPL,
    'ΔWMTSR': ΔWMTSR,
    'ΔWMTSL': ΔWMTSL,
    'ΔWMTS': ΔWMTS,
    'WMPPRGR': WMPPRGR,
    'WMPPRGL': WMPPRGL,
    'WMPRGR': WMPRGR,
    'WMPRGL': WMPRGL,
    'WMPRG': WMPRG,
})

wm_change_womac_df = align_to_index(index, wm_change_womac_df)

# -- merge by ID, READPRJ, and SIDE -- #
wm_change_womac_df = moaks_shared_womac[['ID', 'READPRJ', 'SIDE']].merge(
    wm_change_womac_df,
    left_index=True,
    right_index=True,
    how='left'
)

wm_change_womac = OAI_DataFrame(wm_change_womac_df)

wm_change_womac_df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/wm_change_womac.csv", index=False)