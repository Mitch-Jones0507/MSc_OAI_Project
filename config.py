import pandas as pd
import numpy as np
import os
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame, KL_DataFrame, WOMAC_DataFrame
from modules.classes.oai_dataframe import OAI_DataFrame
from modules.handle_data import handle_mbm_within_grades

# ========= Setting OS ========== #

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'oai_data')

# ========== Initialising OAI data ========== #
# df = pd.Dataframe()

# -- baseline MOAKS dataset -- #
v00_moaks_df = pd.read_sas(os.path.join(data_dir, 'kmri_sq_moaks_bicl00.sas7bdat'))
# -- 12-month MOAKS dataset -- #
v01_moaks_df = pd.read_sas(os.path.join(data_dir, 'kmri_sq_moaks_bicl01.sas7bdat'))

# -- baseline K-L dataset -- #
v00_kl_df = pd.read_sas(os.path.join(data_dir, 'kxr_sq_bu00.sas7bdat'))
# -- 12-month K-L dataset -- #
v01_kl_df = pd.read_sas(os.path.join(data_dir, 'kxr_sq_bu01.sas7bdat'))
# -- 24-month K-L dataset -- #
v03_kl_df = pd.read_sas(os.path.join(data_dir, 'kxr_sq_bu03.sas7bdat'))
# -- 72-month K-L dataset --#
v05_kl_df = pd.read_sas(os.path.join(data_dir, 'kxr_sq_bu05.sas7bdat'))

# -- 12-month WOMAC dataset -- #
v01_womac_df = pd.read_sas(os.path.join(data_dir, 'allclinical01.sas7bdat'))
# -- 24-month WOMAC dataset -- #
v03_womac_df = pd.read_sas(os.path.join(data_dir, 'allclinical03.sas7bdat'))

# ========== Instantiating DataFrame objects ========== #
# Superclass: OAI_DataFrame from modules.classes.oai_dataframe
# Subclasses: MOAKS_DataFrame, KL_DataFrame, WOMAC_DataFrame from modules.classes.moaks_kl_womac_dataframes

# -- baseline MOAKS dataset object -- #
v00_moaks = MOAKS_DataFrame(v00_moaks_df)
# -- 12-month MOAKS dataset object -- #
v01_moaks = MOAKS_DataFrame(v01_moaks_df)

# -- baseline K-L dataset object -- #
v00_kl = KL_DataFrame(v00_kl_df)
# -- 12-month K-L dataset object -- #
v01_kl = KL_DataFrame(v01_kl_df)
# -- 24-month K-L dataset object -- #
v03_kl = KL_DataFrame(v03_kl_df)
# -- 72-month K-L dataset object -- #
v05_kl = KL_DataFrame(v05_kl_df)

# -- 12-month WOMAC dataset object -- #
v01_womac = WOMAC_DataFrame(v01_womac_df)
# -- 24-month WOMAC dataset object -- #
v03_womac = WOMAC_DataFrame(v03_womac_df)

# ========== Calculating sets of unique cases ========== #
# set = set()
# ID = patient identifier, int()
# SIDE = right (1.0) or left (2.0) knee, float()
# READPRJ/reader = clinician identifier, float()

# -- baseline MOAKS ID, SIDE, reader set (5117 knees) -- #
v00_moaks_set = v00_moaks.unique_id(with_side=True, with_reader=True)
# -- 12-month MOAKS ID, SIDE, reader set (3937 knees) -- #
v01_moaks_set = v01_moaks.unique_id(with_side=True, with_reader=True)
# -- baseline MOAKS ID, SIDE set (reader switched off due to different readers between moaks and kl) (3962 knees) -- #
v00_moaks_set_no_reader = v00_moaks.unique_id(with_side=True, with_reader=False)
# -- 12-month MOAKS ID, SIDE set (reader switched off due to different readers between moaks and kl) (3937 knees) -- #
v01_moaks_set_no_reader = v01_moaks.unique_id(with_side=True, with_reader=False)
# -- baseline MOAKS ID set (reader and side switched off due to no side and no readers in womac data) (3229 knees) -- #
v00_moaks_set_no_side_no_reader = v00_moaks.unique_id(with_side=False, with_reader=False)
# -- #
v01_moaks_set_no_side_no_reader = v01_moaks.unique_id(with_side=False, with_reader=False)

# -- baseline K-L ID, SIDE set (reader switched off due to different readers between moaks and kl) (16666 knees) -- #
v00_kl_set = v00_kl.unique_id(with_side=True, with_reader=True)
# -- 12-month K-L shared ID, SIDE set (reader switched off due to different readers between moaks and kl) (8440 knees) -- #
v01_kl_set = v01_kl.unique_id(with_side=True, with_reader=False)
# -- 24-month K-L shared ID, SIDE set (reader switched off due to different readers between moaks and kl) (7988 knees) -- #
v03_kl_set = v03_kl.unique_id(with_side=True, with_reader=False)
# -- 72-month  K-L shared ID, SIDE set (reader switched off due to different readers between moaks and kl) (7632) knees) -- #
v05_kl_set = v05_kl.unique_id(with_side=True, with_reader=False)

# -- 12-month WOMAC shared ID set (reader and side switched off due to no side and no readers in womac data) (4796 knees) -- #
v01_womac_set = v01_womac.unique_id(with_side=False, with_reader=False)
# -- 24-month WOMAC shared ID set (reader and side switched off due to no side and no readers in womac data) (4796 knees) -- #
v03_womac_set = v03_womac.unique_id(with_side=False, with_reader=False)


# ========== Calculating shared sets of unique cases between DataFrames ========== #
# shared_knees = set()
# shared_knees_reader = set()

# -- shared knees, readers between v00_moaks and v01_moaks (3077 knees) -- #
shared_knees_reader = v00_moaks_set.intersection(v01_moaks_set)

# -- shared knees between v00_moaks and v01_kl (3757 knees) -- #
v00_v01_shared_knees_kl = v00_moaks_set_no_reader.intersection(v01_kl_set)
# -- shared knees between v00_moaks and v03_kl (3630 knees) -- #
v00_v03_shared_knees_kl = v00_moaks_set_no_reader.intersection(v03_kl_set)
# -- shared knees between v00_moaks and v05_kl (3481 knees) -- #
v00_v05_shared_knees_kl = v00_moaks_set_no_reader.intersection(v05_kl_set)

# -- shared patients between v00_moaks and v03_womac (3229 patients)
v00_v03_shared_knees_womac = v00_moaks_set_no_side_no_reader.intersection(v03_womac_set)
# -- shared patients between v00_moaks, v01_moaks, v01_womac, and v03_womac (2608 patients)
v00_v03_shared_knees_womac_moaks = v00_moaks_set_no_side_no_reader.intersection(v01_moaks_set_no_side_no_reader, v01_womac_set, v03_womac_set)

# ========== Reducing DataFrames to unique cases only ========== #
# df = DataFrame.__getitem__ = pd.Dataframe()
# womac logic discontinued due to new merge logic

# -- baseline MOAKS dataset with shared ID, SIDE, reader set with v01_moaks (3077 knees) -- #
v00_moaks_shared_df = v00_moaks[:, shared_knees_reader].copy()
# -- 12-month MOAKS dataset with shared ID, SIDE, reader set with v00_moaks (3077 knees) -- #
v01_moaks_shared_df = v01_moaks[:, shared_knees_reader].copy()
# -- baseline MOAKS dataset with shared ID, SIDE set with v01_kl (3757 knees) -- #
v00_moaks_shared_kl_df = v00_moaks[:, v00_v01_shared_knees_kl].drop_duplicates(subset=['ID', 'SIDE']).sort_values(['ID', 'SIDE']).reset_index(drop=True).copy()
# -- baseline MOAKS dataset with shared ID set with v03_womac (3229 patients) -- #
#v00_moaks_shared_womac_df = v00_moaks[:,v00_v03_shared_knees_womac].drop_duplicates(subset=['ID']).sort_values(['ID']).reset_index(drop=True).copy()
# -- 12-month MOAKS dataset with shared ID set with v03_womac (2608 patients) -- #
#v01_moaks_shared_womac_df = v01_moaks[:, v00_v03_shared_knees_womac].drop_duplicates(subset=['ID']).sort_values(['ID']).reset_index(drop=True).copy()

# -- baseline K-L dataset with shared ID, SIDE set with v00_moaks (3757 knees) -- #
v00_kl_shared_moaks_df = v00_kl[:, v00_v01_shared_knees_kl].drop_duplicates(subset=['ID', 'SIDE']).sort_values(['ID', 'SIDE']).reset_index(drop=True).copy()
# -- 12-month K-L dataset with shared ID, SIDE set with v00_moaks (3757 knees) -- #
v01_kl_shared_moaks_df = v01_kl[:, v00_v01_shared_knees_kl].drop_duplicates(subset=['ID', 'SIDE']).sort_values(['ID', 'SIDE']).reset_index(drop=True).copy()
# -- 24-month K-L dataset with shared ID, SIDE set with v00_moaks (3630 knees) -- #
v03_kl_shared_moaks_df = v03_kl[:,v00_v03_shared_knees_kl].drop_duplicates(subset=['ID', 'SIDE']).sort_values(['ID', 'SIDE']).reset_index(drop=True).copy()
# -- 72-month K-L dataset with shared ID, SIDE set with v00_moaks (3481 knees) -- #
v05_kl_shared_moaks_df = v05_kl[:,v00_v05_shared_knees_kl].drop_duplicates(subset=['ID', 'SIDE']).sort_values(['ID', 'SIDE']).reset_index(drop=True).copy()

# -- 12-month WOMAC dataset with shared ID set with v00_moaks (3229 patients) -- #
#v01_womac_shared_moaks_df = v01_womac[:, v00_v03_shared_knees_womac].drop_duplicates(subset=['ID']).sort_values(['ID']).reset_index(drop=True).copy()
# -- 24-month WOMAC dataset with shared ID set with v00_moaks (3229 patients) -- #
#v03_womac_shared_moaks_df = v03_womac[:,v00_v03_shared_knees_womac].drop_duplicates(subset=['ID']).sort_values(['ID']).reset_index(drop=True).copy()

# ========== Merging v00_moaks, v01_moaks, v01_kl, v05_kl ========== #

v00_moaks_shared_kl_df_id_side_reader = pd.DataFrame(
    list(
        v00_moaks_set_no_reader
        .intersection(v01_moaks_set_no_reader, v01_kl_set, v03_kl_set)
    ),
    columns=['ID','SIDE']
)

v00_moaks_kl_mbm = v00_moaks_df[['ID', 'SIDE'] + [column for column in v00_moaks_df.columns if 'MBM' in column]]
v01_moaks_kl_mbm = v01_moaks_df[['ID', 'SIDE'] + [column for column in v01_moaks_df.columns if 'MBM' in column]]

v01_moaks_kl_kl = v01_kl_df[['ID','SIDE'] + [column for column in v01_kl_df.columns if 'KL' in column]]
v03_moaks_kl_kl = v03_kl_df[['ID','SIDE'] + [column for column in v03_kl_df.columns if 'KL' in column]]

moaks_shared_kl_df = (
    v00_moaks_shared_kl_df_id_side_reader
    .merge(v00_moaks_kl_mbm, on=['ID', 'SIDE'], how='left')
    .merge(v01_moaks_kl_mbm, on=['ID', 'SIDE'], how='left')
    .merge(v01_moaks_kl_kl, on=['ID','SIDE'], how='left')
    .merge(v03_moaks_kl_kl, on=['ID','SIDE'], how='left')
    .copy()
)
moaks_shared_kl_df = moaks_shared_kl_df.copy()

# -- update mbm scoring to handle 0.5 (within-grade worsening) and -0.5 (within-grade improvement) cases -- #
moaks_shared_kl_df = handle_mbm_within_grades(moaks_shared_kl_df)

# -- save to csv -- #
moaks_shared_kl_df.to_csv('/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/moaks_shared_kl.csv', index=False)

# ========== Merging v00_moaks, v01_moaks, v01_womac, v03_womac ========== #

# -- create new dataframe with only ID, SIDE, READPRJ common to v00_moaks and v01_moaks, and IDs common to v01_womac and v03_womac -- #
v00_moaks_shared_womac_df_id_side_reader = \
    pd.DataFrame(list(shared_knees_reader),columns=['ID','SIDE','READPRJ'])
v00_moaks_shared_womac_df_id_side_reader = \
    v00_moaks_shared_womac_df_id_side_reader[v00_moaks_shared_womac_df_id_side_reader['ID'].isin(v00_v03_shared_knees_womac_moaks)]

v00_moaks_womac_mbm = v00_moaks_df[['ID', 'SIDE','READPRJ'] + [column for column in v00_moaks_df.columns if 'MBM' in column]]
v01_moaks_womca_mbm = v01_moaks_df[['ID', 'SIDE','READPRJ'] + [column for column in v01_moaks_df.columns if 'MBM' in column]]

v01_moaks_womac_wom = v01_womac_df[['ID'] + [column for column in v01_womac_df.columns if 'WOM' in column]]
v03_moaks_womca_wom = v03_womac_df[['ID'] + [column for column in v03_womac_df.columns if 'WOM' in column]]

# -- merge v00 and v01 moaks columns by unique ID, SIDE, READPRJ -- #
moaks_shared_womac_df = v00_moaks_shared_womac_df_id_side_reader \
    .merge(v00_moaks_womac_mbm, on=['ID', 'READPRJ', 'SIDE'], how='left') \
    .merge(v01_moaks_womca_mbm, on=['ID', 'READPRJ', 'SIDE'], how='left')

# -- merge v01 and v03 womac columns by unique IDs -- #
moaks_shared_womac_df = moaks_shared_womac_df \
    .merge(v01_moaks_womac_wom, on=['ID'], how='left') \
    .merge(v03_moaks_womca_wom, on=['ID'], how='left')

# -- access necessary columns for analysis -- #
#moaks_shared_womac_df = (moaks_shared_womac_df[['ID','READPRJ','SIDE','V00MBMSFMA','V00MBMNFMA','V00MBMPFMA','V00MBMSFMC',
            #'V00MBMNFMC','V00MBMPFMC','V00MBMSFMP','V00MBMNFMP','V00MBMPFMP','V01MBMSFMA','V01MBMNFMA','V01MBMPFMA',
            #'V01MBMSFMC','V01MBMNFMC','V01MBMPFMC','V01MBMSFMP','V01MBMNFMP','V01MBMPFMP','V00MBMSTMA','V00MBMNTMA',
            #'V00MBMPTMA','V00MBMSTMC','V00MBMNTMC','V00MBMPTMC','V00MBMSTMP','V00MBMNTMP','V00MBMPTMP','V01MBMSTMA',
            #'V01MBMNTMA','V01MBMPTMA','V01MBMSTMC','V01MBMNTMC','V01MBMPTMC','V01MBMSTMP','V01MBMNTMP','V01MBMPTMP',
            #'V01WOMKPR','V01WOMKPL','V03WOMKPR','V03WOMKPL','V01WOMTSR','V01WOMTSL','V03WOMTSR','V03WOMTSL']]
            #.sort_values(['ID','READPRJ','SIDE']).copy())

# -- copy before NaN imputation -- #
moaks_shared_womac_df = moaks_shared_womac_df.copy()

# -- fill NaNs with each variable's median imputation to preserve skewness and sample size -- #
moaks_shared_womac_df_columns = [column for column in moaks_shared_womac_df.columns if column not in \
                                 ['ID','READPRJ','SIDE']]
#for column in v00_moaks_shared_womac_df_columns:
    #moaks_shared_womac_df[column] = moaks_shared_womac_df[column].fillna(moaks_shared_womac_df[column].median())

# -- update mbm scoring to handle 0.5 (within-grade worsening) and -0.5 (within-grade improvement) cases -- #
moaks_shared_womac_df = handle_mbm_within_grades(moaks_shared_womac_df)

# -- save to csv -- #
moaks_shared_womac_df.to_csv("/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/moaks_shared_womac.csv", index=False)

# ========== Instantiating shared DataFrame objects ==========#

# -- baseline MOAKS dataset object with shared ID, SIDE, reader set with v01_moaks (3077 knees) -- #
v00_moaks_shared = MOAKS_DataFrame(v00_moaks_shared_df)
# -- 12-month MOAKS dataset object with shared ID, SIDE, reader set with v01_moaks (3077 knees) -- #
v01_moaks_shared = MOAKS_DataFrame(v01_moaks_shared_df)
# -- baseline MOAKS dataset object with shared ID, SIDE set with v01-kl (3775 knees) -- #
v00_moaks_shared_kl = MOAKS_DataFrame(v00_moaks_shared_kl_df)

# -- baseline K-L dataset object with shared ID, SIDE set with v01_moaks (3775 knees) -- #
v00_kl_shared_moaks = KL_DataFrame(v00_kl_shared_moaks_df)
# -- 12-month K-L dataset object with shared, ID, SIDE set with v00_moaks (3775 knees) -- #
v01_kl_shared_moaks = KL_DataFrame(v01_kl_shared_moaks_df)
# -- 24-month K-L dataset object with shared, ID, SIDE set with v00_moaks (3630 knees) -- #
v03_kl_shared_moaks = KL_DataFrame(v03_kl_shared_moaks_df)
# -- 72-month K-L dataset object with shared, ID, SIDE set with v00_moaks (3481 knees) -- #
v05_kl_shared_moaks = KL_DataFrame(v05_kl_shared_moaks_df)

# -- merged and condensed v00_moaks (MOAKS, WOMAC) dataset object with shared ID, SIDE, reader set with v01_moaks, and shared ID set with v01_womac and v03_womac (3077 knees, reader) -- #
moaks_shared_womac = OAI_DataFrame(moaks_shared_womac_df)
