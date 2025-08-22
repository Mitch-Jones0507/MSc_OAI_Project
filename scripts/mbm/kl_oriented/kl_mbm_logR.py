import pandas as pd
import numpy as np
from modules.machine_learning import run_multinomial_l1_logistic
from scripts.mbm.kl_oriented.kl_mbm_relationships import v01_kl_drop, v03_kl_drop, Δkl_drop, v00_v01_kl_drop_lasso_coef_df, \
    v00_v03_kl_drop_lasso_coef_df, v01_v03_kl_drop_lasso_coef_df, v00_kl_drop_change_lasso_coef_df, v01_kl_drop_change_lasso_coef_df
from scripts.mbm.kl_oriented.mbm_var_relationships_kl import v00_moaks_shared_kl_drop_mbm_df, v01_moaks_shared_kl_drop_mbm_df

v01_kl_drop = v01_kl_drop.copy()
v03_kl_drop = v03_kl_drop.copy()
Δkl_drop = Δkl_drop.copy()

# ========== Define MBM columns from PLS ========== #

v00_v01_moaks_lasso_nonzero_features = v00_v01_kl_drop_lasso_coef_df.loc[
    (v00_v01_kl_drop_lasso_coef_df['Coefficient'].abs() > 1e-9) &
    (v00_v01_kl_drop_lasso_coef_df['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v00_v01_moaks_shared_kl_drop_mbm_lasso_df = v00_moaks_shared_kl_drop_mbm_df[v00_v01_moaks_lasso_nonzero_features]

v00_v03_moaks_lasso_nonzero_features = v00_v03_kl_drop_lasso_coef_df.loc[
    (v00_v03_kl_drop_lasso_coef_df['Coefficient'].abs() > 1e-9) &
    (v00_v03_kl_drop_lasso_coef_df['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v00_v03_moaks_shared_kl_drop_mbm_lasso_df = v00_moaks_shared_kl_drop_mbm_df[v00_v03_moaks_lasso_nonzero_features]

v01_v03_moaks_lasso_nonzero_features = v01_v03_kl_drop_lasso_coef_df.loc[
    (v01_v03_kl_drop_lasso_coef_df['Coefficient'].abs() > 1e-9) &
    (v01_v03_kl_drop_lasso_coef_df['Feature'] != 'Intercept'),
    'Feature'
].tolist()
v01_v03_moaks_shared_kl_drop_mbm_lasso_df = v01_moaks_shared_kl_drop_mbm_df[v01_v03_moaks_lasso_nonzero_features]

v01_kl_drop_categorised = v01_kl_drop.map(lambda x: str(x) if x < 3 else '3.0-4.0').astype('category')
v03_kl_drop_categorised = v03_kl_drop.map(lambda x: str(x) if x < 3 else '3.0-4.0').astype('category')
Δkl_drop_categorised = Δkl_drop.map(lambda x: 'No Change' if x == 0 else 'Negative Change' if -4 <= x < 0 else 'Positive Change').astype('category')

v00_v01_moaks_kl_model, v00_v01_moaks_kl_metrics, v00_v01_moaks_kl_coefs_df = run_multinomial_l1_logistic(v00_moaks_shared_kl_drop_mbm_df, v01_kl_drop_categorised, Cs=np.logspace(-6, 6, 20))
v00_v01_report = v00_v01_moaks_kl_metrics['classification_report']
v00_v01_confusion_matrix = v00_v01_moaks_kl_metrics['confusion_matrix']
v00_v01_confusion_matrix = pd.DataFrame(v00_v01_confusion_matrix,index=[f"True {i}" for i in range(len(v00_v01_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v00_v01_confusion_matrix))])

v00_v03_moaks_kl_model, v00_v03_moaks_kl_metrics, v00_v03_moaks_kl_coefs_df = run_multinomial_l1_logistic(v00_moaks_shared_kl_drop_mbm_df, v03_kl_drop_categorised, Cs=np.logspace(-6, 6, 20))
v00_v03_report = v00_v03_moaks_kl_metrics['classification_report']
v00_v03_confusion_matrix = v00_v03_moaks_kl_metrics['confusion_matrix']
v00_v03_confusion_matrix = pd.DataFrame(v00_v03_confusion_matrix,index=[f"True {i}" for i in range(len(v00_v03_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v00_v03_confusion_matrix))])

v01_v03_moaks_kl_model, v01_v03_moaks_kl_metrics, v01_v03_moaks_kl_coefs_df = run_multinomial_l1_logistic(v01_moaks_shared_kl_drop_mbm_df, v03_kl_drop_categorised, Cs=np.logspace(-6, 6, 20))
v01_v03_report = v01_v03_moaks_kl_metrics['classification_report']
v01_v03_confusion_matrix = v01_v03_moaks_kl_metrics['confusion_matrix']
v01_v03_confusion_matrix = pd.DataFrame(v01_v03_confusion_matrix,index=[f"True {i}" for i in range(len(v01_v03_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v01_v03_confusion_matrix))])

v00_v01_moaks_kl_lasso_model, v00_v01_moaks_kl_lasso_metrics, v00_v01_moaks_kl_lasso_coefs_df = run_multinomial_l1_logistic(v00_v01_moaks_shared_kl_drop_mbm_lasso_df, v01_kl_drop_categorised, Cs=np.logspace(-2, 2, 20))
v00_v01_lasso_report = v00_v01_moaks_kl_lasso_metrics['classification_report']
v00_v01_lasso_confusion_matrix = v00_v01_moaks_kl_lasso_metrics['confusion_matrix']
v00_v01_lasso_confusion_matrix = pd.DataFrame(v00_v01_lasso_confusion_matrix,index=[f"True {i}" for i in range(len(v00_v01_lasso_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v00_v01_lasso_confusion_matrix))])

v00_v03_moaks_kl_lasso_model, v00_v03_moaks_kl_lasso_metrics, v00_v03_moaks_kl_lasso_coefs_df = run_multinomial_l1_logistic(v00_v03_moaks_shared_kl_drop_mbm_lasso_df, v03_kl_drop_categorised, Cs=np.logspace(-2, 2, 20))
v00_v03_lasso_report = v00_v03_moaks_kl_lasso_metrics['classification_report']
v00_v03_lasso_confusion_matrix = v00_v03_moaks_kl_lasso_metrics['confusion_matrix']
v00_v03_lasso_confusion_matrix = pd.DataFrame(v00_v03_lasso_confusion_matrix,index=[f"True {i}" for i in range(len(v00_v03_lasso_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v00_v03_lasso_confusion_matrix))])

v01_v03_moaks_kl_lasso_model, v01_v03_moaks_kl_lasso_metrics, v01_v03_moaks_kl_lasso_coefs_df = run_multinomial_l1_logistic(v01_v03_moaks_shared_kl_drop_mbm_lasso_df, v03_kl_drop_categorised, Cs=np.logspace(-2, 2, 20))
v01_v03_lasso_report = v01_v03_moaks_kl_lasso_metrics['classification_report']
v01_v03_lasso_confusion_matrix = v01_v03_moaks_kl_lasso_metrics['confusion_matrix']
v01_v03_lasso_confusion_matrix = pd.DataFrame(v01_v03_lasso_confusion_matrix,index=[f"True {i}" for i in range(len(v01_v03_lasso_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v01_v03_lasso_confusion_matrix))])

