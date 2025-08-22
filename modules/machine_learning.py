from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, \
classification_report, roc_curve, auc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.multioutput import RegressorChain
from xgboost import XGBClassifier, XGBRegressor
import plotly.graph_objects as go
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

    y_encoded = y.astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

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

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    class_report = classification_report(y_test, y_pred, output_dict=True)

    all_classes = sorted(y_encoded.unique())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_classes)

    gain_importance = pd.DataFrame({
        'Feature': X.columns,
        'GainImportance': model.feature_importances_
    }).sort_values(by='GainImportance', ascending=False)

    perm_importance = permutation_importance(model, X_test, y_test,
                                             n_repeats=10, random_state=random_state)
    perm_df = pd.DataFrame({
        'Feature': X.columns,
        'PermImportance': perm_importance.importances_mean,
        'PermImportance_std': perm_importance.importances_std
    }).sort_values(by='PermImportance', ascending=False)

    feature_df = gain_importance.merge(perm_df, on='Feature')

    metrics = {
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

    return model, metrics, feature_df

def run_xgboost_with_threshold(X, y, threshold=0.5, test_size=0.2, random_state=42):

    # --- Convert string labels to integers if necessary ---
    if y.dtype == 'O' or y.dtype.name.startswith('category'):
        unique_labels = sorted(y.unique())
        label_map = {v: i for i, v in enumerate(unique_labels)}
        y_numeric = y.map(label_map)
    else:
        y_numeric = y

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=test_size, random_state=random_state, stratify=y_numeric
    )

    # --- Fit XGBoost ---
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # --- Feature importance ---
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # --- Classification report ---
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # --- ROC for multi-class ---
    classes = np.unique(y_numeric)
    if len(classes) == 2:
        # Binary case: apply threshold
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_thresh = (y_proba >= threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    else:
        # Multi-class case: One-vs-Rest, macro and micro
        y_test_bin = label_binarize(y_test, classes=classes)
        y_proba_bin = model.predict_proba(X_test)

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba_bin.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        # Macro-average
        aucs = []
        fig = go.Figure()
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_bin[:, i])
            auc_cls = auc(fpr, tpr)
            aucs.append(auc_cls)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Class {cls} (AUC={auc_cls:.2f})"))
        auc_macro = np.mean(aucs)

        # Add micro-average
        fig.add_trace(go.Scatter(x=fpr_micro, y=tpr_micro, mode="lines",
                                 name=f"Micro-average ROC (AUC={auc_micro:.2f})", line=dict(dash="dot", width=3)))
        # Chance line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))

        fig.update_layout(
            title=f"(Macro AUC={auc_macro:.2f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )

    return coef_df, report_dict, fig


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
