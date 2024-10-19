import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.feature_selection import RFE
from joblib import Parallel, delayed
import numpy as np


# Load dataset function
def LoadDataset(df):
    print("Available columns in the dataset:", df.columns)
    return df.columns

# Start training with optimized models and prediction accuracy
def StartTraining(target_column, df):
    # Drop rows with missing values
    df = df.dropna()

    # Remove the target column from the list of features
    df_features = df.drop(target_column, axis=1)

    # List of all features
    filtered_columns = df_features.columns.tolist()

    print(f"\nInitial Features: {filtered_columns}")

    # Perform multiple models and evaluate with 20% prediction accuracy
    results = select_features_with_Mutual_Information(filtered_columns, target_column, df)

    return results  # Return the results list

def select_features_with_Mutual_Information(features_list, target, df):
    # Available models
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'SVR': SVR(kernel='linear'),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=3)
    }

    # Parallelize feature selection and model evaluation
    results = Parallel(n_jobs=-1)(delayed(evaluate_model_with_feature_importance)(model_name, model, df, features_list, target)
                                   for model_name, model in models.items())

    return results  # Return the list of results

# Helper function to perform feature selection, train, and evaluate each model
def evaluate_model_with_feature_importance(model_name, model, df, features, target):
    X = df[features]
    y = df[target]

    # Split the dataset (80% train, 20% for testing/prediction)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_test)

    # Calculate the R² score on test set
    r2 = r2_score(y_test, y_pred_train)

    # Use Mutual Information for non-tree-based models
    if model_name in ['RandomForest', 'GradientBoosting']:
        importances = model.feature_importances_
        selected_features = [features[i] for i in range(len(features)) if importances[i] > 0]
        print(f"Model: {model_name}, Feature Importances: {importances}")
    else:
        mi = mutual_info_regression(X_train, y_train)
        selected_features = [features[i] for i in range(len(features)) if mi[i] > 0]
        print(f"Model: {model_name}, Mutual Information Scores: {mi}")

    # Evaluate prediction accuracy with RMSE (Root Mean Squared Error)
    mse = root_mean_squared_error(y_test, y_pred_train)
    rmse = np.sqrt(mse)

    # Calculate the accuracy percentage
    mean_actual = np.mean(y_test)
    accuracy = 100 * (1 - (rmse / mean_actual))

    # Print model, selected features, R² score, RMSE, and accuracy
    print(f"Model: {model_name}, Selected Features: {selected_features}, R² score: {r2:.4f}, RMSE: {rmse:.4f}, Accuracy: {accuracy:.2f}%")

    return model_name, selected_features, r2, rmse, accuracy  # Return model name, features, R² score, RMSE, and accuracy
