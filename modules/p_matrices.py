import pandas as pd
import numpy as np
import re
from statsmodels.stats.multitest import multipletests

def build_p_matrix(regression_df):
    p_matrix = []
    for index, row in regression_df.iterrows():
        formula = row['formula']
        p_values = row['p_values']
        if p_values is None:
            continue
        target = formula.split('~')[0].strip()
        for predictor, p_value in p_values.items():
            if pd.isna(p_value):
                continue
            p_matrix.append({'target': target, 'predictor': predictor, 'p_value': p_value})

    p_matrix_df = pd.DataFrame(p_matrix)

    reject_null, corrected_p_values, _, _ = multipletests(p_matrix_df['p_value'], method='fdr_bh')
    p_matrix_df['p_value_bh'] = corrected_p_values
    median_p_matrix_df = (p_matrix_df.groupby(['target', 'predictor'])['p_value_bh'].median().unstack(fill_value=np.nan))

    return pd.DataFrame(median_p_matrix_df)

def sort_p_matrix(p_matrix):

    index = sorted(p_matrix.index, key = lambda row: (
        0 if 'MBMS' in row else
        1 if 'MBMN' in row else
        2 if 'MBMP' in row else 3
    ))

    without_intercept = [column for column in p_matrix.columns if 'Intercept' not in column]
    columns = sorted(without_intercept, key = lambda column: (
        3 if (':' in column or '*' in column) else
        1 if re.search(r'\*\*\s*2', column) else
        2 if re.search(r'\*\*\s*3', column) else
        0,
        0 if 'MBMS' in column else
        1 if 'MBMN' in column else
        2 if 'MBMP' in column else
        3

    ))

    p_matrix = p_matrix.loc[index,columns].T
    p_matrix.index.name = None
    p_matrix.columns.name = None

    return p_matrix