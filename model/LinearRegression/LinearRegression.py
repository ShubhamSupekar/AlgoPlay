# with stepwise forward selection mwthod for feature selection 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# df = pd.read_csv('Datasets\housing.csv')

# df = pd.read_csv('Datasets\Customer Purchasing Behaviors.csv')


def LoadDataset(df):
    # df = pd.read_csv('Datasets/laptop_price - dataset.csv')
    # Step 1: Ask the user to select the target column
    print("Available columns in the dataset:", df.columns)
    return df.columns


def StartTraining(target_column,df):
# target_column = input("Enter the target column (the column to predict): ")
# Step 2: Automatically select numerical columns as potential features
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove the target column from the list of features
    numerical_columns.remove(target_column)

    # Step 3: Calculate the correlation matrix and apply the threshold
    correlation_matrix = df[numerical_columns].corr().abs()

    # Step 4: Set a threshold for correlation (for example, 0.8)
    threshold = 0.8
    filtered_columns = numerical_columns.copy()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > threshold:
                colname = correlation_matrix.columns[i]
                if colname in filtered_columns:
                    print(f"Removing highly correlated feature: {colname}")
                    filtered_columns.remove(colname)

    print(f"\nFiltered Features after applying correlation threshold: {filtered_columns}")
    best_features, best_r2 = forward_selection(filtered_columns.copy(), target_column)
    return filtered_columns, best_features, best_r2

# Stepwise forward selection function
def forward_selection(features_list, target):
    selected_features = []
    best_r2 = -float('inf')
    current_best_r2 = 0
    while len(features_list) > 0:
        temp_r2_scores = []
        for feature in features_list:
            combo = selected_features + [feature]
            X_train, X_test, y_train, y_test = train_test_split(df[combo], df[target], test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            temp_r2_scores.append((combo, r2))
        
        # Find the best new feature to add
        best_combo, best_r2 = max(temp_r2_scores, key=lambda x: x[1])
        
        if best_r2 > current_best_r2:
            current_best_r2 = best_r2
            selected_features = best_combo
            features_list.remove(best_combo[-1])  # Remove the best feature from available features
        else:
            break  # If no improvement, stop
    
    return selected_features, current_best_r2

# Stepwise feature selection

# Output the result
# print(f"\nBest combination of features: {best_features}, with an R² score of: {best_r2}")
