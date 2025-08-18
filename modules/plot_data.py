import numpy as np
import pandas as pd
import patsy as ps
import re
import plotly.io as pl
import plotly.express as px
import plotly.graph_objects as go
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
pl.templates.default = "plotly_dark"

def plot_predictor_target_data(dataframe):
    columns = list(dataframe.columns)
    moaks_df_counted = dataframe.groupby(columns).size().reset_index(name='count')

    predictor_target_scatter = px.scatter(
        moaks_df_counted,
        x=columns[0],
        y=columns[1],
        size='count',
        color=columns[2] if len(columns) > 2 else None,
        size_max=40,
        opacity=0.8
    )

    # Improve layout
    predictor_target_scatter.update_layout(
        xaxis=dict(title=columns[0], showgrid=False),
        yaxis=dict(title=columns[1], showgrid=False),
    )

    return predictor_target_scatter


def plot_moaks_data(moaks_dataframe: MOAKS_DataFrame):

    columns = list(moaks_dataframe.df.columns)
    moaks_df_counted = moaks_dataframe.df.groupby(columns).size().reset_index(name='count')

    moaks_df_scatter = px.scatter_3d(moaks_df_counted, x = columns[0], y = columns[1], z = columns[2], size = 'count',
                                     size_max = 40, opacity = 0.8)
    moaks_df_scatter.update_layout(scene=dict(
        xaxis=dict(title=columns[0], showgrid = False),
        yaxis=dict(title=columns[1], showgrid = False),
        zaxis=dict(title=columns[2], showgrid = False),
    ))

    return moaks_df_scatter

def handle_regression(moaks_df: MOAKS_DataFrame, formula_df: pd.DataFrame, equation: int):

    formula = formula_df.loc[equation,'formula']
    left, right = formula.split('~')
    dependent = left.strip()
    terms = [term.strip() for term in right.split('+')]
    independents = list(dict.fromkeys(
        var
        for term in terms
        for var in re.split(r'\*\*|:|\*', term.replace('I(', '').replace(')', '').strip())
        if not var.isnumeric()  # ← Filter out things like '3'
    ))

    y = moaks_df[dependent]
    x = moaks_df[independents[0]]
    x_label = independents[0]
    if len(independents) > 1:
        z = moaks_df[independents[1]]
        z_label = independents[1]
    else:
        remainder = [column for column in moaks_df.df.columns if column not in [dependent, independents[0],'count']]
        z = moaks_df[remainder[0]]
        z_label = remainder[0]

    coefficients = formula_df.loc[equation,'params'].to_dict()

    x_range = np.linspace(x.min(), x.max(), 50)
    z_range = np.linspace(z.min(), z.max(), 50)
    X, Z = np.meshgrid(x_range, z_range)

    x_flat = X.ravel()
    z_flat = Z.ravel()
    grid = pd.DataFrame({x_label: x_flat, z_label: z_flat})

    X_design = ps.dmatrix(right.strip(), grid, return_type='dataframe')
    beta = np.array([coefficients.get(col, 0) for col in X_design.columns])
    y_pred_flat = X_design @ beta
    Y = y_pred_flat.values.reshape(X.shape)

    return X, Y, Z

def plot_moaks_data_regression_surface(moaks_scatter, moaks_df: MOAKS_DataFrame, formula_df: pd.DataFrame, equation: int):

    formula = formula_df.loc[equation, 'formula']
    X, Y, Z = handle_regression(moaks_df, formula_df, equation)

    moaks_regression_surface = go.Figure(data = moaks_scatter)
    moaks_regression_surface.add_trace(go.Surface(
        x = np.clip(X, 0, 12),
        y = np.clip(Y,0,12),
        z = np.clip(Z,0,12),
        opacity = 0.3,
        colorscale='YlGnBu',
        showscale = False,
    ))
    moaks_regression_surface.update_layout(
        title = dict(
            text = formula,
            x = 0.5,
            xanchor = 'center',
        )
    )

    return moaks_regression_surface

