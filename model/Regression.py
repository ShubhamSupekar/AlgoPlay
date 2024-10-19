import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from joblib import Parallel, delayed



# Load dataset function
def LoadDataset(df):
    print("Available columns in the dataset:", df.columns)
    return df.columns



# Start training with optimized models and RFE for feature selection
def StartTraining(target_column, df):
    # Drop rows with missing values
    df = df.dropna()

    # Remove the target column from the list of features
    df_features = df.drop(target_column, axis=1)

    # List of all features
    filtered_columns = df_features.columns.tolist()

    print(f"\nInitial Features: {filtered_columns}")

    # Perform RFE with multiple models
    results = select_features_with_Mutual_Information(filtered_columns, target_column, df)

    return results  # Return the results list




def select_features_with_Mutual_Information(features_list, target, df):
    results = []  # Store results in a list

    # Available models, excluding PyTorch Neural Network
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



# Helper function to perform feature selection and evaluate each model
def evaluate_model_with_feature_importance(model_name, model, df, features, target):
    X = df[features]
    y = df[target]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model for scikit-learn models
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate the R² score
    r2 = r2_score(y_test, y_pred)

    if model_name in ['RandomForest', 'GradientBoosting']:
        # Feature importance for tree-based models
        importances = model.feature_importances_
        selected_features = [features[i] for i in range(len(features)) if importances[i] > 0]  # Use all important features

        # Print feature importance
        print(f"Model: {model_name}, Feature Importances: {importances}")
    else:
        # Use Mutual Information for non-tree-based models
        mi = mutual_info_regression(X_train, y_train)
        selected_features = [features[i] for i in range(len(features)) if mi[i] > 0]  # Use features with positive MI

        # Print Mutual Information scores
        print(f"Model: {model_name}, Mutual Information Scores: {mi}")

    # Print the model, selected features, and R² score
    print(f"Model: {model_name}, Selected Features: {selected_features}, R² score: {r2:.4f}")

    return model_name, selected_features, r2  # Return results