import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def LoadDataset(df):
    print("Available columns in the dataset:", df.columns)
    return df.columns

def StartTraining(target_column, df):
    # Remove the target column from the list of features
    df_features = df.drop(target_column, axis=1)

    # Step 3: Calculate the correlation matrix and apply the threshold
    correlation_matrix = df_features.corr().abs()

    # Step 4: Set a threshold for correlation (for example, 0.8)
    threshold = 0.8
    filtered_columns = df_features.columns.tolist()  # Use a list for filtering

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] > threshold:
                colname = correlation_matrix.columns[i]
                if colname in filtered_columns:
                    print(f"Removing highly correlated feature: {colname}")
                    filtered_columns.remove(colname)

    print(f"\nFiltered Features after applying correlation threshold: {filtered_columns}")
    best_features, best_r2 = forward_selection(filtered_columns.copy(), target_column, df)
    return best_features, best_r2

# Stepwise forward selection function
def forward_selection(features_list, target, df):
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

