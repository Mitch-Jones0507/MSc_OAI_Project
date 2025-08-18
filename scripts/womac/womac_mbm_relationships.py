import pandas as pd
from config import moaks_shared_womac
from modules.plot_data import plot_predictor_target_data
from scripts.mbm.womac_oriented.mbm_load import mbm_load_womac
from scripts.womac.womac_change import wm_change_womac

left_moaks_data_df = mbm_load_womac.df[mbm_load_womac.df['SIDE'] == 1]
right_moaks_data_df = mbm_load_womac.df[mbm_load_womac.df['SIDE'] == 2]

right_moaks_change_womac_df = pd.merge(
    right_moaks_data_df,
    wm_change_womac.df,
    on=['ID','READPRJ', 'SIDE'],
    how='inner',)

left_moaks_change_womac_df = pd.merge(
    left_moaks_data_df,
    wm_change_womac.df,
    on=['ID','READPRJ', 'SIDE'],
    how='inner',
)

right_moaks_change_womac_df = right_moaks_change_womac_df.drop(['WMPPRGL','WMPRGL','ΔWMKPL','ΔWMTSL'],axis=1)
left_moaks_change_womac_df = left_moaks_change_womac_df.drop(['WMPPRGR','WMPRGR','ΔWMKPR','ΔWMTSR'],axis=1)

right_moaks_change_womac_df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/right_moaks_change_womac.csv", index=False)
left_moaks_change_womac_df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/left_moaks_change_womac.csv", index=False)

print(right_moaks_change_womac_df)
print(left_moaks_change_womac_df)

predictor_target_data_df = left_moaks_change_womac_df[['ΔMBMNTFMLOADSUM','WMPPRGL']]

scatter = plot_predictor_target_data(predictor_target_data_df)
scatter.show()