# import necessary libraries
import joblib
import numpy as np
import pandas as pd
from pyarrow import nulls
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt



# Convert the 'date' column to datetime
def date_time(downtime):
    downtime['Date'] = pd.to_datetime(downtime['Date'], dayfirst=True)

    # Extract year, month, and day into separate columns
    downtime['year'] = downtime['Date'].dt.year
    downtime['month'] = downtime['Date'].dt.month
    downtime['day'] = downtime['Date'].dt.day

    return downtime

# handle cyclic features end of year
def month_cyclic(downtime):
    downtime['month_sin'] = np.sin(2 * np.pi * downtime['month'] / 12)
    downtime['month_cos'] = np.cos(2 * np.pi * downtime['month'] / 12)

    return downtime

# drop all unnecessary columns
def drop_cols(downtime):
    downtime = downtime.drop(['Date', 'year', 'month', 'day'], axis = 1)

    return downtime

# process data for EDA
def prep_for_eda(downtime):
    functions = [date_time, month_cyclic, drop_cols]

    for func in functions:
        downtime = func(downtime)

    return downtime

# split into predictors and target
def split_data(df):
    X = df.loc[:,df.columns != 'Downtime']

    # machine and assemble line id have a 1:1 correlation: drop 1 of them
    X = X.drop("Assembly_Line_No", axis = 1)

    y = df['Downtime']

    return X, y

# split downtime by machine type
def split_by_machine_type(downtime):
    downtime_l1 = downtime.loc[downtime['Machine_ID'] == "Makino-L1-Unit1-2013"]
    downtime_l2 = downtime.loc[downtime['Machine_ID'] == "Makino-L2-Unit1-2015"]
    downtime_l3 = downtime.loc[downtime['Machine_ID'] == "Makino-L3-Unit1-2015"]

    # downtime_l1 = downtime_l1.drop("machine_type", axis = 1)
    # downtime_l2 = downtime_l2.drop("machine_type", axis=1)
    # downtime_l3 = downtime_l3.drop("machine_type", axis=1)

    return downtime_l1, downtime_l2, downtime_l3


# Models
models = {
    "CatBoost": CatBoostClassifier(cat_features = ['Machine_ID'], random_state = 42, verbose = 0),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state = 42),
    "RandomForest": RandomForestClassifier(random_state = 42)
}

# Model Search Params
search_spaces = {
    "CatBoost": {
        "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
        "depth": Integer(3, 15),
        "iterations": Integer(50, 500)
    },
    "XGBoost": {
        "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
        "max_depth": Integer(3, 15),
        "n_estimators": Integer(50, 500),
        "colsample_bytree": Real(0.5, 1.0)
    },
    "RandomForest": {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 20),
        "min_samples_split": Integer(2, 20),
        "max_features": Integer(2, 8)
    }
}


# return metrics for model
def get_best_params_and_score(model_name, X, y):
    X_adj = X.copy()
    y_adj = y.copy()
    if model_name in models:
        print(f"Optimizing {model_name}...")
        opt = BayesSearchCV(
            models[model_name],
            search_spaces[model_name],
            n_iter=50,  # Number of iterations for Bayesian Optimization
            cv=5,
            scoring="f1",  # Choose F1-score as primary metric for optimization
            n_jobs=-1,  # Use all available cores
            verbose=2
        )

        le = LabelEncoder()
        # handle Machine ID in RandomForest if it's present
        if model_name in ("RandomForest", "XGBoost"):

            if "Machine_ID" in X_adj.columns:
                # transform categorical
                X_adj["Machine_ID"] = le.fit_transform(X_adj["Machine_ID"])
                print(dict(zip(le.classes_, le.transform(le.classes_))))

        # transform label
        y_adj = le.fit_transform(y)
        print(dict(zip(le.classes_, le.transform(le.classes_))))

        joblib.dump(X_adj, f'outputs/X_{model_name}.pkl')
        joblib.dump(y_adj, f'outputs/y_{model_name}.pkl')

        opt.fit(X_adj, y_adj)

        # Evaluate the best model on all metrics
        best_model = opt.best_estimator_

        # save the best model and adjusted tables as needed
        joblib.dump(best_model, f'outputs/Best{model_name}.pkl')

        # use cross_validate to return metrics for the best model
        cv_results = cross_validate(best_model, X_adj, y_adj, cv = 5, scoring = ('accuracy', 'balanced_accuracy', 'f1'))

        metrics = {
            "Best Parameters": opt.best_params_,
            # "Best F1-Score (CV)": opt.best_score_,
            "Best Model Average Test Accuracy": cv_results["test_accuracy"].mean(),
            "Best Model Average Test Balanced Accuracy": cv_results["test_balanced_accuracy"].mean(),
            "Best Model Average Test F1-Score": cv_results["test_f1"].mean()
        }

        joblib.dump(metrics, f'outputs/metrics{model_name}.pkl')

        # Print results
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics, best_model, X_adj, y_adj
    else:
        print(f"Model {model_name} not found in models or search_spaces.")


# calculate feature importances for each model
def feature_importance(model_name, X, y, best_model):
    best_model.fit(X, y)

    # Check model type and retrieve feature importances
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        raise ValueError("Model does not support feature importances.")

    # Create a DataFrame of feature names and their importances
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Print feature importances
    print("Feature Importances (sorted):")
    print(feature_importances)

    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color='skyblue')
    plt.title(f"Feature Importance: {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"./outputs/feature_importances_{model_name}.png")
    plt.show()

    return feature_importances

# return metrics for the independent machien models

# return metrics for model
def get_score_diff_machines(model_name, best_model, X, y):
    X_adj = X.copy()
    y_adj = y.copy()
    if model_name in models:

        le = LabelEncoder()
        # handle Machine ID in RandomForest if it's present
        if model_name in ("RandomForest", "XGBoost"):
            if "Machine_ID" in X_adj.columns:
                # transform categorical
                X_adj["Machine_ID"] = le.fit_transform(X_adj["Machine_ID"])
                # print(dict(zip(le.classes_, le.transform(le.classes_))))

        # transform label
        y_adj = le.fit_transform(y)
        print(dict(zip(le.classes_, le.transform(le.classes_))))

        # use cross_validate to return metrics for the best model
        cv_results = cross_validate(best_model, X_adj, y_adj, cv = 5, scoring = ('accuracy', 'balanced_accuracy', 'f1'))

        metrics = {
            "Best Model Average Test Accuracy": cv_results["test_accuracy"].mean(),
            "Best Model Average Test Balanced Accuracy": cv_results["test_balanced_accuracy"].mean(),
            "Best Model Average Test F1-Score": cv_results["test_f1"].mean()
        }

        # Print results
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics, X_adj, y_adj
    else:
        print(f"Model {model_name} not found in models or search_spaces.")

# calculate feature importances for each model
def feature_importance_diff_machines(model_name, X, y, best_model, machine_name):
    best_model.fit(X, y)

    # Check model type and retrieve feature importances
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        raise ValueError("Model does not support feature importances.")

    # Create a DataFrame of feature names and their importances
    feature_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Print feature importances
    print("Feature Importances (sorted):")
    print(feature_importances)

    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color='skyblue')
    plt.title(f"Feature Importance: {model_name}-{machine_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"./outputs/feature_importances_{model_name}_{machine_name}.png")
    plt.show()