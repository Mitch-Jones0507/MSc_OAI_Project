from missing_imputation import moaks_shared_womac_drop_df, moaks_shared_womac_median_df
from modules.handle_data import handle_vectors_womac
import plotly.express as px
import pingouin as pg

moaks_shared_womac_drop_df = moaks_shared_womac_drop_df.copy()
moaks_shared_womac_median_df = moaks_shared_womac_median_df.copy()

moaks_shared_womac_drop_df['ΔWOMADLL'] = moaks_shared_womac_drop_df['V03WOMADLL'] - moaks_shared_womac_drop_df['V01WOMADLL']
moaks_shared_womac_drop_df['ΔWOMADLR'] = moaks_shared_womac_drop_df['V03WOMADLR'] - moaks_shared_womac_drop_df['V01WOMADLR']
moaks_shared_womac_drop_df['ΔWOMKPL'] = moaks_shared_womac_drop_df['V03WOMKPL'] - moaks_shared_womac_drop_df['V01WOMKPL']
moaks_shared_womac_drop_df['ΔWOMKPR'] = moaks_shared_womac_drop_df['V03WOMKPR'] - moaks_shared_womac_drop_df['V01WOMKPR']
moaks_shared_womac_drop_df['ΔWOMSTFL'] = moaks_shared_womac_drop_df['V03WOMSTFL'] - moaks_shared_womac_drop_df['V01WOMSTFL']
moaks_shared_womac_drop_df['ΔWOMSTFR'] = moaks_shared_womac_drop_df['V03WOMSTFR'] - moaks_shared_womac_drop_df['V01WOMSTFR']

moaks_shared_womac_median_df['ΔWOMADLL'] = moaks_shared_womac_median_df['V03WOMADLL'] - moaks_shared_womac_median_df['V01WOMADLL']
moaks_shared_womac_median_df['ΔWOMADLR'] = moaks_shared_womac_median_df['V03WOMADLR'] - moaks_shared_womac_median_df['V01WOMADLR']
moaks_shared_womac_median_df['ΔWOMKPL'] = moaks_shared_womac_median_df['V03WOMKPL'] - moaks_shared_womac_median_df['V01WOMKPL']
moaks_shared_womac_median_df['ΔWOMKPR'] = moaks_shared_womac_median_df['V03WOMKPR'] - moaks_shared_womac_median_df['V01WOMKPR']
moaks_shared_womac_median_df['ΔWOMSTFL'] = moaks_shared_womac_median_df['V03WOMSTFL'] - moaks_shared_womac_median_df['V01WOMSTFL']
moaks_shared_womac_median_df['ΔWOMSTFR'] = moaks_shared_womac_median_df['V03WOMSTFR'] - moaks_shared_womac_median_df['V01WOMSTFR']

v01_womac_drop_vectors_left = handle_vectors_womac(moaks_shared_womac_drop_df, 'V01WOMADLL','V01WOMKPL','V01WOMSTFL')
v01_womac_drop_vectors_right = handle_vectors_womac(moaks_shared_womac_drop_df, 'V01WOMADLR','V01WOMKPR','V01WOMSTFR')
v03_womac_drop_vectors_left = handle_vectors_womac(moaks_shared_womac_drop_df, 'V03WOMADLL','V03WOMKPL','V03WOMSTFR')
v03_womac_drop_vectors_right = handle_vectors_womac(moaks_shared_womac_drop_df, 'V03WOMADLR','V03WOMKPR','V03WOMSTFR')

v01_womac_median_vectors_left = handle_vectors_womac(moaks_shared_womac_median_df, 'V01WOMADLL','V01WOMKPL','V01WOMSTFL')
v01_womac_median_vectors_right = handle_vectors_womac(moaks_shared_womac_median_df, 'V01WOMADLR','V01WOMKPR','V01WOMSTFR')
v03_womac_median_vectors_left = handle_vectors_womac(moaks_shared_womac_median_df, 'V03WOMADLL','V03WOMKPL','V03WOMSTFR')
v03_womac_median_vectors_right = handle_vectors_womac(moaks_shared_womac_median_df, 'V03WOMADLR','V03WOMKPR','V03WOMSTFR')

womac_drop_vectors_left_change = handle_vectors_womac(moaks_shared_womac_drop_df, 'ΔWOMADLL','ΔWOMKPL', 'ΔWOMSTFL')
womac_drop_vectors_right_change = handle_vectors_womac(moaks_shared_womac_drop_df, 'ΔWOMADLR','ΔWOMKPR','ΔWOMSTFR')

womac_median_vectors_left_change = handle_vectors_womac(moaks_shared_womac_median_df, 'ΔWOMADLL','ΔWOMKPL', 'ΔWOMSTFL')
womac_median_vectors_right_change = handle_vectors_womac(moaks_shared_womac_median_df,'ΔWOMADLR','ΔWOMKPR','ΔWOMSTFR')

v01_womac_drop_vectors_left_chart = px.bar(v01_womac_drop_vectors_left,x='Vector',y='Count',log_y=True,title='12-Month WOMAC Drop Vectors (Left Knee)')
v01_womac_drop_vectors_right_chart = px.bar(v01_womac_drop_vectors_right,x='Vector',y='Count',log_y=True,title='12-Month WOMAC Drop Vectors (Right Knee)')
v03_womac_drop_vectors_left_chart = px.bar(v03_womac_drop_vectors_left,x='Vector',y='Count',log_y=True,title='24-Month WOMAC Drop Vectors (Left Knee)')
v03_womac_drop_vectors_right_chart = px.bar(v03_womac_drop_vectors_right,x='Vector',y='Count',log_y=True, title='24-Month WOMAC Drop Vectors (Right Knee)')

v01_columns = moaks_shared_womac_drop_df[['V01WOMADLL','V01WOMKPL','V01WOMSTFL']].dropna()
multivariate_normality_v01 = pg.multivariate_normality(v01_columns, alpha=.05)

v03_columns = moaks_shared_womac_drop_df[['V03WOMADLL','V03WOMKPL','V03WOMSTFL']].dropna()
multivariate_normality_v03 = pg.multivariate_normality(v03_columns, alpha=.05)
