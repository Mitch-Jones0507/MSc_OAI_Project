import statsmodels.api as sm
from mbm_load import *
from collections import Counter
import itertools as it
from itertools import combinations

def run_wls(dataframe, formulas):
    results = {}
    for category, formulae in formulas.items():
        results[category] = {}
        for formula in formulae:
            model_result = dataframe.wls(formula)
            results[category][formula] = model_result.params, model_result.pvalues, model_result.rsquared
    return results

# -- shared baseline mbm load dataframe -- #
v00_moaks_shared_mbm_load_df = v00_moaks_shared[['V00MBMSFMLOAD', 'V00MBMNFMLOAD', 'V00MBMPFMLOAD']]

# -- shared baseline mbm load object -- #
v00_moaks_shared_mbm_load = MOAKS_DataFrame(v00_moaks_shared_mbm_load_df)

# -- predictor and outcome variable list -- #
mbm_space = v00_moaks_shared_mbm_load.get_variables("V00MBM").columns.tolist()

# -- regression formulae structure for wls -- #
formulas = {
    'linear': [],
    'quadratic': [],
    'cubic': [],
    'interaction': [],
}

# -- populating formulas dictionary with regression modes -- #
for y in mbm_space:
    others = [variable for variable in mbm_space if variable != y]
    for x, z in combinations(others, 2):
        xz_sorted = sorted([x, z])
        x, z = xz_sorted[0], xz_sorted[1]
        # -- linear -- #
        formulas['linear'].append(f'{y} ~ {x}')
        formulas['linear'].append(f'{y} ~ {x} + {z}')
        # -- quadratic -- #
        formulas['quadratic'].append(f'{y} ~ {x} + I({x}**2)')
        formulas['quadratic'].append(f'{y} ~ {x} + {z} + I({x}**2) + I({z}**2)')
        # -- cubic -- #
        formulas['cubic'].append(f'{y} ~ {x} + I({x}**2) + I({x}**3)')
        formulas['cubic'].append(f'{y} ~ {x} + {z} + I({x}**2) + I({z}**2) + I({x}**3) + I({z}**3)')
        # -- interaction -- #
        formulas['interaction'].append(f'{y} ~ {x}:{z}')
        formulas['interaction'].append(f'{y} ~ {x}*{z}')

# -- mbms load, mbmn load, and mbmp load regression summary dictionary -- #
mbm_relationships = run_wls(v00_moaks_shared_mbm_load, formulas)

# -- mbms load, mbmn load, and mbmn load regression summary dataframe -- #
mbm_relationships_df = pd.DataFrame([
    {'type': category, 'formula': formula, 'params': params, 'p_values': p_values, 'r_squared': r_squared}
    for category, formulas in mbm_relationships.items()
    for formula, (params, p_values, r_squared) in formulas.items()
])

# -- adding to csv -- #
mbm_relationships_df.to_csv('/Users/mitch/NDA/Core_Resources/Dissertation_Project/data/mbm_relationships.csv', index=False)
