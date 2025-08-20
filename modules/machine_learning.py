from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.multioutput import RegressorChain
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np

scaler = StandardScaler()

def run_multinomial_l1_logistic(X: pd.DataFrame, y: pd.Series,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                Cs = None, cv: int = 5):

    y_encoded = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if Cs is None:
        Cs = np.logspace(-4, 4, 20)

    model = LogisticRegressionCV(
        Cs=Cs,
        penalty='l1',
        solver='saga',
        cv=cv,
        max_iter=5000,
        random_state=random_state,
        scoring='f1_weighted'
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    coef_df = pd.DataFrame(model.coef_, columns=X.columns)
    coef_df['Class'] = model.classes_
    coef_df = coef_df.melt(id_vars='Class', var_name='Feature', value_name='Coefficient')

    metrics = {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

    return model, metrics, coef_df

def run_hist_gb_classifier(X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2,
                           random_state: int = 42,
                           max_depth=None,
                           learning_rate=0.1,
                           max_iter=100):

    y_encoded = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    model = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    perm_importance = permutation_importance(model, X_test, y_test,
                                             n_repeats=10, random_state=random_state)
    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean,
        'Importance_std': perm_importance.importances_std
    }).sort_values(by='Importance', ascending=False)

    metrics = {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

    return model, metrics, feature_df

def run_xgb_classifier(X: pd.DataFrame, y: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       max_depth: int = 6,
                       learning_rate: float = 0.1,
                       n_estimators: int = 200,
                       subsample: float = 0.85,
                       colsample_bytree: float = 0.85):

    # Encode target
    y_encoded = y.astype('category').cat.codes

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Base XGBoost model with fixed hyperparameters
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(y_encoded.unique()),
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)

    all_classes = sorted(y_encoded.unique())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_classes)

    # Gain importance
    gain_importance = pd.DataFrame({
        'Feature': X.columns,
        'GainImportance': model.feature_importances_
    }).sort_values(by='GainImportance', ascending=False)

    # Permutation importance
    perm_importance = permutation_importance(model, X_test, y_test,
                                             n_repeats=10, random_state=random_state)
    perm_df = pd.DataFrame({
        'Feature': X.columns,
        'PermImportance': perm_importance.importances_mean,
        'PermImportance_std': perm_importance.importances_std
    }).sort_values(by='PermImportance', ascending=False)

    # Merge feature importance
    feature_df = gain_importance.merge(perm_df, on='Feature')

    metrics = {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

    return model, metrics, feature_df

def run_xgb_multioutput_regressor(X: pd.DataFrame, y: pd.DataFrame,
                                  test_size: float = 0.2,
                                  random_state: int = 42,
                                  max_depth: int = 6,
                                  learning_rate: float = 0.1,
                                  n_estimators: int = 200,
                                  subsample: float = 0.85,
                                  colsample_bytree: float = 0.85):
    X, y = X.align(y, join="inner", axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    xgb = XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        objective='reg:squarederror'
    )

    model = RegressorChain(xgb, order='random', random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {}
    for i, col in enumerate(y.columns):
        metrics[col] = {
            'r2': r2_score(y_test.iloc[:, i], y_pred[:, i]),
            'mse': mean_squared_error(y_test.iloc[:, i], y_pred[:, i]),
            'mae': mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        }

    gain_importance_list = []

    for i, est in enumerate(model.estimators_):
        features = X.columns.tolist() + [f'PrevTarget_{j}' for j in range(i)]
        df = pd.DataFrame({
            'Feature': features,
            'GainImportance': est.feature_importances_,
            'Target': y.columns[i]
        })
        gain_importance_list.append(df)

    gain_importance_df = pd.concat(gain_importance_list, ignore_index=True)

    gain_importance_df = gain_importance_df.sort_values(by=['Target', 'GainImportance'], ascending=[True, False])

    perm_importance = permutation_importance(model, X_test, y_test,
                                             n_repeats=10, random_state=random_state,
                                             scoring='r2')
    perm_df = pd.DataFrame({
        'Feature': X.columns,
        'PermImportance': perm_importance.importances_mean,
        'PermImportance_std': perm_importance.importances_std
    }).sort_values(by='PermImportance', ascending=False)

    feature_df = gain_importance_df.merge(perm_df, on='Feature')

    return model, metrics, feature_df
