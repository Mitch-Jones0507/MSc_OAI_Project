import numpy as np
import pandas as pd

def handle_mbm_within_grades(dataframe):

    for column in [column for column in dataframe if 'V01MBM' in column]:
        v00_column = column.replace('V01MBM','V00MBM')

        dataframe[column] = np.where(
            dataframe[column] == 0.5,
            dataframe[v00_column] + 0.5,
        np.where(
            dataframe[column] == -0.5,
            dataframe[v00_column] -0.5,
            dataframe[column]
            )
        )

    return dataframe

def handle_vectors_womac(dataframe, series_x, series_y, series_z):
    vector_dataframe = pd.DataFrame()

    for variable in [series_x, series_y, series_z]:
        min_val = int(dataframe[variable].min())
        max_val = int(dataframe[variable].max())

        # Bin by exact integer scores, handle NaN safely
        vector_dataframe[variable] = pd.cut(
            dataframe[variable],
            bins=range(min_val, max_val + 2),
            labels=range(min_val, max_val + 1),
            include_lowest=True
        )

        # Replace NaN with -1 safely
        vector_dataframe[variable] = vector_dataframe[variable].cat.codes.replace(-1, -1)

    # Count occurrences of each vector
    vector_dataframe_count = (
        vector_dataframe.groupby([series_x, series_y, series_z])
        .size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=False)
    )

    # String representation
    vector_dataframe_count['Vector'] = vector_dataframe_count.apply(
        lambda row: f"({row[series_x]},{row[series_y]},{row[series_z]})", axis=1
    )

    return vector_dataframe_count
