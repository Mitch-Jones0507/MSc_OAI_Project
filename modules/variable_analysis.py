import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm

scaler = StandardScaler()

def full_regression_analysis(moaks_dataframe):

    variable_space = moaks_dataframe.get_variables('',fm_only=False).columns.tolist()
    formulae = build_regression_formulae(variable_space)
    relationships = run_wls_with_formulae(moaks_dataframe, formulae)
    results = build_regression_summary_dataframe(relationships)

    return results

def build_regression_formulae(variable_space):

    formulae = {
        'linear': [],
        'quadratic': [],
        'cubic': [],
        'interaction': []
    }

    for y in variable_space:
        leftover_variables = [variable for variable in variable_space if variable != y]

        if len(leftover_variables) == 1:
            x = leftover_variables[0]

            # -- linear terms -- #
            formulae['linear'].append(f'{y} ~ {x}')

            # -- quadratic terms -- #
            formulae['quadratic'].append(f'{y} ~ I({x}**2)')
            formulae['quadratic'].append(f'{y} ~ {x} + I({x}**2)')

            # -- cubic terms -- #
            formulae['cubic'].append(f'{y} ~ I({x}**3)')
            formulae['cubic'].append(f'{y} ~ {x} + I({x}**2) + I({x}**3)')

        else:
            for x, z in combinations(leftover_variables, 2):
                x, z = sorted([x, z])

                # -- linear terms -- #
                formulae['linear'].append(f'{y} ~ {x}')
                formulae['linear'].append(f'{y} ~ {x} + {z}')

                # -- quadratic terms -- #
                formulae['quadratic'].append(f'{y} ~ I({x}**2)')
                formulae['quadratic'].append(f'{y} ~ {x} + I({x}**2)')
                formulae['quadratic'].append(f'{y} ~ {x} + {z} + I({x}**2) + I({z}**2)')
                formulae['quadratic'].append(f'{y} ~ {x} + I({z}**2)')

                # -- cubic terms -- #
                formulae['cubic'].append(f'{y} ~ I({x}**3)')
                formulae['cubic'].append(f'{y} ~ {x} + I({x}**2) + I({x}**3)')
                formulae['cubic'].append(f'{y} ~ {x} + {z} + I({x}**2) + I({z}**2) + I({x}**3) + I({z}**3)')
                formulae['cubic'].append(f'{y} ~ {x} + I({z}**2) + I({z}**3)')

                # -- interaction terms -- #
                formulae['interaction'].append(f'{y} ~ {x}:{z}')
                formulae['interaction'].append(f'{y} ~ {x}*{z}')
                formulae['interaction'].append(f'{y} ~ {x}*I({z}**2)')
                formulae['interaction'].append(f'{y} ~ {x}*I({z}**3)')

    return formulae

def run_wls_with_formulae(moaks_dataframe: MOAKS_DataFrame, formulas):

    results = {}
    for category, formulae in formulas.items():
        results[category] = {}
        for formula in formulae:
            model_result = moaks_dataframe.wls(formula)
            results[category][formula] = model_result.params, model_result.pvalues, model_result.rsquared
    return results

def build_regression_summary_dataframe(summary_dictionary):

    summary_dataframe = pd.DataFrame([
        {'type': category, 'formula': formula, 'params': params, 'p_values': p_values, 'r_squared': r_squared}
        for category, formulas in summary_dictionary.items()
        for formula, (params, p_values, r_squared) in formulas.items()
    ])

    return summary_dataframe

def cronbach_alpha(moaks_dataframe: MOAKS_DataFrame):

    item_variance = moaks_dataframe.df.var(axis=0, ddof=1)
    total_variance = moaks_dataframe.df.sum(axis=1).var(ddof=1)
    n_items = moaks_dataframe.df.shape[1]
    return (n_items / (n_items - 1)) * (1 - (item_variance.sum() / total_variance))

def factor_analysis(dataframe, loadings=None):

    columns_to_drop = [column for column in ['ID', 'SIDE', 'READPRJ'] if column in dataframe.columns]
    df = dataframe.drop(columns=columns_to_drop).copy()

    x_standardised = scaler.fit_transform(df)
    if loadings is not None:
        factor_score = x_standardised @ loadings
        factor_loadings = loadings
    else:
        factor_analyser = FactorAnalyzer(n_factors=1,rotation=None)
        factor_analyser.fit(x_standardised)
        factor_score = factor_analyser.transform(x_standardised)
        factor_loadings = factor_analyser.loadings_

    factor_df = pd.DataFrame(factor_score, index=df.index, columns=['factor'])
    return factor_df.squeeze(), factor_loadings

def pca(dataframe, pca_model=None, components=1, initial_statistics=None):

    columns_to_drop = [column for column in ['ID', 'SIDE', 'READPRJ'] if column in dataframe.columns]
    df = dataframe.drop(columns=columns_to_drop).copy()

    x = df.values

    if pca_model is None:
        x_standardised = scaler.fit_transform(x)

        pca_model = PCA(n_components=components)
        pca_model.fit(x_standardised)
        pc = pca_model.transform(x_standardised)

        initial_statistics = {'mean': scaler.mean_, 'standard_deviation': np.sqrt(scaler.var_)}
    else:
        x_standardised = (x - initial_statistics['mean']) / initial_statistics['standard_deviation']
        pc = x_standardised @ pca_model.components_.T

    pc_df = pd.DataFrame(data=pc, index=df.index, columns=[f'PC{i+1}' for i in range(components)])

    return pc_df.squeeze(), pca_model, initial_statistics

def align_to_index(index, dataframe):
    aligned = pd.concat([index, dataframe], axis=1)
    return aligned.iloc[:, len(index.columns):].reset_index(drop=True)

def run_linear_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()

    coef_df = pd.DataFrame({
        "Feature": ["Intercept"] + X.columns.tolist(),
        "Coefficient": [model.intercept_] + model.coef_.tolist(),
        "p_value": ols_model.pvalues.values
    }).sort_values(by="Coefficient", key=abs, ascending=False).reset_index(drop=True)

    results = {
        "R2": r2,
        "RMSE": rmse,
        "Intercept": model.intercept_,
    }

    return model, results, coef_df

def run_multiple_linear_regression(X: pd.DataFrame, Y, test_size: float = 0.2, random_state: int = 42):

    if isinstance(Y, pd.Series):
        Y = Y.to_frame(name=Y.name or "Target")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    r2_scores = {col: r2_score(Y_test[col], Y_pred[:, i]) for i, col in enumerate(Y.columns)}
    rmse_scores = {col: np.sqrt(mean_squared_error(Y_test[col], Y_pred[:, i])) for i, col in enumerate(Y.columns)}

    coef_list = []
    for i, col in enumerate(Y.columns):
        X_sm = sm.add_constant(X)
        ols_model = sm.OLS(Y[col], X_sm).fit()
        coef_df = pd.DataFrame({
            "Feature": ["Intercept"] + X.columns.tolist(),
            "Coefficient": [model.intercept_[i]] + model.coef_[i].tolist(),
            "p_value": ols_model.pvalues.values
        })
        coef_df["Target"] = col
        coef_list.append(coef_df)

    coef_df_full = pd.concat(coef_list, ignore_index=True)

    results = {
        "R2": r2_scores,
        "RMSE": rmse_scores,
        "Intercept": dict(zip(Y.columns, model.intercept_)),
    }

    return model, results, coef_df_full

def create_transformed_features(X: pd.DataFrame, degree: int = 2, interaction_only: bool = False):
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly_array = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(X_poly_array, columns=feature_names, index=X.index)

def run_multiple_pls(X: pd.DataFrame, Y,
                     n_components: int = 2,
                     test_size: float = 0.2,
                     random_state: int = 42):

    if isinstance(Y, pd.Series):
        Y = Y.to_frame(name=Y.name or "Target")

    combined = pd.concat([X, Y], axis=1)
    combined_clean = combined.dropna(axis=0, how='any')
    X_clean = combined_clean[X.columns].reset_index(drop=True)
    Y_clean = combined_clean[Y.columns].reset_index(drop=True)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_clean, Y_clean, test_size=test_size, random_state=random_state
    )

    n_components_safe = min(n_components, X_train.shape[0] - 1, X_train.shape[1], Y_train.shape[1])

    model = PLSRegression(n_components=n_components_safe, scale=True)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    r2_scores = {}
    rmse_scores = {}
    for i, col in enumerate(Y.columns):
        r2_scores[col] = r2_score(Y_test[col], Y_pred[:, i])
        rmse_scores[col] = np.sqrt(mean_squared_error(Y_test[col], Y_pred[:, i]))

    W = model.x_weights_
    P = model.x_loadings_
    C = model.y_loadings_
    coef_scaled = W @ np.linalg.pinv(P.T @ W) @ C.T

    X_mean = X_clean.mean()
    X_std = X_clean.std()
    Y_mean = Y_clean.mean()
    Y_std = Y_clean.std()

    coef_unscaled = coef_scaled * Y_std.values[None, :] / X_std.values[:, None]
    intercepts = Y_mean.values - X_mean.values @ coef_unscaled

    coef_list = []
    for i, col in enumerate(Y.columns):
        coef_df = pd.DataFrame({
            "Feature": ["Intercept"] + X.columns.tolist(),
            "Coefficient": [intercepts[i]] + coef_unscaled[:, i].tolist(),
            "Target": col
        })
        coef_list.append(coef_df)

    coef_df_full = pd.concat(coef_list, ignore_index=True)

    X_latent_df = pd.DataFrame(model.x_scores_, columns=[f'PLS Component {i + 1}' for i in range(n_components_safe)],
                               index=X_train.index)

    results = {
        "R2": r2_scores,
        "RMSE": rmse_scores,
        "n_components": n_components_safe
    }

    return model, results, coef_df_full, X_latent_df

def run_lasso_regression(X: pd.DataFrame, Y: pd.DataFrame,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        alphas=np.logspace(-2, 2, 30)):

    if isinstance(Y, pd.Series):
        Y = Y.to_frame(name=Y.name or "Target")

    combined = pd.concat([X, Y], axis=1)
    combined_clean = combined.dropna(axis=0, how='any')
    X_clean = combined_clean[X.columns].reset_index(drop=True)
    Y_clean = combined_clean[Y.columns].reset_index(drop=True)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_clean, Y_clean, test_size=test_size, random_state=random_state
    )

    model = make_pipeline(
        StandardScaler(),
        MultiTaskLassoCV(alphas=alphas, cv=5, random_state=random_state, max_iter=20000)
    )
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    r2_scores = {col: r2_score(Y_test[col], Y_pred[:, i]) for i, col in enumerate(Y.columns)}
    rmse_scores = {col: mean_squared_error(Y_test[col], Y_pred[:, i])**0.5 for i, col in enumerate(Y.columns)}

    lasso_model = model.named_steps['multitasklassocv']

    coef_matrix = lasso_model.coef_.T

    coef_df = pd.DataFrame(coef_matrix, columns=Y.columns, index=X.columns).reset_index()
    coef_df = coef_df.melt(id_vars='index', var_name='Target', value_name='Coefficient').rename(
        columns={'index': 'Feature'})

    intercept_df = pd.DataFrame({
        'Feature': ['Intercept'] * Y.shape[1],
        'Target': Y.columns.tolist(),
        'Coefficient': lasso_model.intercept_.tolist()
    })

    coef_df_full = pd.concat([intercept_df, coef_df], ignore_index=True)

    results = {
        "R2": r2_scores,
        "RMSE": rmse_scores
    }

    return model, results, coef_df_full