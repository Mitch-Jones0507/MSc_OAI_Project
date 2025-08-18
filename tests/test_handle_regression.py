import unittest
import numpy as np
import pandas as pd
from modules.classes.moaks_kl_womac_dataframes import MOAKS_DataFrame
from modules.plot_data import handle_regression

class TestHandleRegression(unittest.TestCase):

    def setUp(self):

        data = {
            'x': np.linspace(0,10,10),
            'y': np.linspace(0,5,10),
            'z': np.linspace(0,20,10),
            'count': [1]*10
        }
        df = pd.DataFrame(data)
        self.moaks_df = MOAKS_DataFrame(df)

        formulas = {
            'formula': ['y ~ I(x**3)', 'y ~ x + z + I(x**2) + I(z**2) + I(x**3) + I(z**3)'],
            'params': [pd.Series({'Intercept':1, 'I(x**3)': 2}), pd.Series({'Intercept':0.004, 'x': 0.56, 'z': 0.024,
                    'I(x**2)':0.034, 'I(z**2)':0.004,'I(x**3)':0.035,'I(z**3)':0.0023})]
        }
        self.formula_df = pd.DataFrame(formulas)
        self.X, self.Y, self.Z = handle_regression(self.moaks_df, self.formula_df, 1)

    def test_shapes(self):
        self.assertEqual(self.X.shape, self.Y.shape)
        self.assertEqual(self.X.shape, self.Z.shape)