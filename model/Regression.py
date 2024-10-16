import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch-based Neural Network for regression
class NeuralNetworkRegressor(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    results = select_features_with_rfe(filtered_columns, target_column, df)

    return results  # Return the results list

# Feature selection using Recursive Feature Elimination (RFE) for different models
def select_features_with_rfe(features_list, target, df):
    results = []  # Store results in a list

    # Available models, including PyTorch Neural Network
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'SVR': SVR(kernel='linear'),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=3),
        'PyTorchNN': NeuralNetworkRegressor
    }

    # Parallelize feature selection and model evaluation
    results = Parallel(n_jobs=-1)(delayed(evaluate_model_with_rfe)(model_name, model, df, features_list, target)
                                   for model_name, model in models.items())

    return results  # Return the list of results

# Helper function to perform RFE and evaluate each model
def evaluate_model_with_rfe(model_name, model, df, features, target):
    X = df[features]
    y = df[target]

    # Perform RFE with the model, skipping for PyTorch (RFE only for sklearn models)
    if model_name != 'PyTorchNN':
        rfe = RFE(estimator=model, n_features_to_select=5)  # Select top 5 features
        X_rfe = rfe.fit_transform(X, y)
    else:
        X_rfe = X.values  # No RFE for PyTorch

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)

    if model_name == 'PyTorchNN':
        # Convert data to PyTorch tensors and use GPU
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).cuda().unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).cuda().unsqueeze(1)

        # Initialize PyTorch model
        model = NeuralNetworkRegressor(X_train.shape[1]).cuda()

        # Print GPU usage
        print("Training PyTorch model on GPU." if torch.cuda.is_available() else "Training PyTorch model on CPU.")

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the PyTorch model
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate the PyTorch model
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy()

        r2 = r2_score(y_test, y_pred)
        selected_features = features  # No RFE applied
    else:
        # Train the model for scikit-learn models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate the R² score
        r2 = r2_score(y_test, y_pred)

        # Get the selected features for scikit-learn models
        selected_features = [features[i] for i in range(len(features)) if rfe.support_[i]]

    # Print the model, selected features, and R² score
    print(f"Model: {model_name}, Selected Features: {selected_features}, R² score: {r2:.4f}")

    return model_name, selected_features, r2  # Return results

# Example usage:
# df = pd.read_csv('your_dataset.csv')
# target_column = 'target_column_name'
# results = StartTraining(target_column, df)
# print(f"\nResults: {results}")  