import pandas as pd

class OAI_DataFrame:

    def __init__(self, df):
        self.df = df
        if 'ID' in self.df.columns:
            df['ID'] = df['ID'].astype(int)
        if 'READPRJ' in self.df.columns:
            self.readprj = 'READPRJ'
        elif 'readprj' in self.df.columns:
            self.readprj = 'readprj'

    def __str__(self) -> str:
        return str(self.df)

    def __getitem__(self, key) -> pd.DataFrame:
        if isinstance(key, tuple) and len(key) == 2:
            cols, rows = key
            row_key = next(iter(rows))
            if isinstance(cols, slice) and cols == slice(None):
                cols = self.df.columns
            if isinstance(row_key, int):
                mask = self.df['ID'].isin(rows)
                return self.df.loc[mask,cols]
            if rows is None or len(row_key) == 0:
                return self.df.loc[[], cols]
            if len(row_key) == 3:
                mask = self.df.apply(lambda row: (row['ID'], row['SIDE'], row[self.readprj]) in rows, axis=1)
                return self.df.loc[mask, cols]
            elif len(row_key) == 2:
                mask = self.df.apply(lambda row: (row['ID'], row['SIDE']) in rows, axis=1)
                return self.df.loc[mask, cols]
            elif len(row_key) == 1:
                mask = self.df.apply(lambda row: (row['ID']) in rows, axis=1)
                return self.df.loc[mask, cols]
        else:
            return self.df[key]

    def __setitem__(self, key, value) -> None:
        self.df[key] = value

    def __len__(self) -> int:
        return self.df.shape[0]

    def unique_id(self, with_side=False, with_reader=False) -> set:

        if with_side and with_reader:
            return set(zip(self.df['ID'], self.df['SIDE'], self.df[self.readprj]))
        elif with_side:
            return set(zip(self.df['ID'], self.df['SIDE']))
        else:
            return set(self.df['ID'])

    def get_variables(self, test, fm_only=True) -> pd.DataFrame:
        if fm_only:
            selected_columns = [col for col in self.df.columns if test in col and 'FM' in col]
        else:
            selected_columns = [col for col in self.df.columns if test in col]
        return self.df[selected_columns]
