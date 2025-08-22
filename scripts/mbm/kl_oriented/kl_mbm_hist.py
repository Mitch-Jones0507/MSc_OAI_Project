import pandas as pd
from modules.machine_learning import run_hist_gb_classifier
from scripts.mbm.kl_oriented.mbm_var_relationships_kl import v00_moaks_shared_kl_drop_mbm_df, v01_moaks_shared_kl_drop_mbm_df
from scripts.mbm.kl_oriented.kl_mbm_logR import v00_v01_moaks_shared_kl_drop_mbm_lasso_df, v00_v03_moaks_shared_kl_drop_mbm_lasso_df, \
v01_v03_moaks_shared_kl_drop_mbm_lasso_df, v01_kl_drop_categorised, v03_kl_drop_categorised, Δkl_drop_categorised

v00_v01_moaks_kl_model, v00_v01_moaks_kl_metrics, v00_v01_moaks_kl_coefs_df = run_hist_gb_classifier(v00_moaks_shared_kl_drop_mbm_df, v01_kl_drop_categorised)
print(v00_v01_moaks_kl_coefs_df)
v00_v01_report = v00_v01_moaks_kl_metrics['classification_report']
v00_v01_confusion_matrix = v00_v01_moaks_kl_metrics['confusion_matrix']
v00_v01_confusion_matrix = pd.DataFrame(v00_v01_confusion_matrix,index=[f"True {i}" for i in range(len(v00_v01_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v00_v01_confusion_matrix))])

v00_v03_moaks_kl_model, v00_v03_moaks_kl_metrics, v00_v03_moaks_kl_coefs_df = run_hist_gb_classifier(v00_moaks_shared_kl_drop_mbm_df, v03_kl_drop_categorised)
v00_v03_report = v00_v03_moaks_kl_metrics['classification_report']
v00_v03_confusion_matrix = v00_v03_moaks_kl_metrics['confusion_matrix']
v00_v03_confusion_matrix = pd.DataFrame(v00_v03_confusion_matrix,index=[f"True {i}" for i in range(len(v00_v03_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v00_v03_confusion_matrix))])

v01_v03_moaks_kl_model, v01_v03_moaks_kl_metrics, v01_v03_moaks_kl_coefs_df = run_hist_gb_classifier(v01_moaks_shared_kl_drop_mbm_df, v03_kl_drop_categorised)
v01_v03_report = v01_v03_moaks_kl_metrics['classification_report']
v01_v03_confusion_matrix = v01_v03_moaks_kl_metrics['confusion_matrix']
v01_v03_confusion_matrix = pd.DataFrame(v01_v03_confusion_matrix,index=[f"True {i}" for i in range(len(v01_v03_confusion_matrix))],columns=[f"Pred {i}" for i in range(len(v01_v03_confusion_matrix))])


