import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from collections import Counter
import patsy as ps
from modules.classes.oai_dataframe import OAI_DataFrame

class MOAKS_DataFrame(OAI_DataFrame):

    def wls(self, formula):
        try:
            y, X = ps.dmatrices(formula, self.df, return_type='dataframe')
        except Exception as e:
            raise ValueError(f"Error parsing formula '{formula}': {e}")
        combined = X.copy()
        combined['__y__'] = y
        points = list(zip(*(combined[col] for col in combined.columns)))
        counts = Counter(points)
        weights = [counts[pt] for pt in points]
        model = sm.WLS(y, X, weights=weights)
        result = model.fit()
        return result

    def wls_validation(self, formula, test_size=0.2):
        train_df, test_df = train_test_split(self.df, test_size = test_size, random_state = None)


class KL_DataFrame(OAI_DataFrame):

    def calculate_progression(self, previous_kl_dataframe, test: str):

        current_col = next((column for column in self.df.columns if test in column), None)
        previous_col = next((column for column in previous_kl_dataframe.df.columns if test in column), None)

        if current_col is None or previous_col is None:
            raise ValueError(f"No column found for test '{test}' in one of the dataframes.")

        difference_series = self.df[current_col] - previous_kl_dataframe.df[previous_col]

        return pd.DataFrame({
            f"{previous_col}": previous_kl_dataframe.df[previous_col],
            f"{current_col}": self.df[current_col],
            "PRG": difference_series
        })

class WOMAC_DataFrame(OAI_DataFrame):

    def womac(self):
        return self.df