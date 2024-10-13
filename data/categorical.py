import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_categorical_columns(df:pd.DataFrame, threshold:int=5):
    columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns=[]
    for i in range(len(columns)):
        if len(df.loc[:,columns[i]].drop_duplicates())<=threshold:
            categorical_columns.append(columns[i])
    return categorical_columns


class Categorical:
    def __init__(self):
        self.dataset=None
        self.encoding_scheme=None
        self.column_name=None
        self.Encoder=None
    def isClassSet(self):
        return any([self.dataset is not None,
                    self.encoding_scheme is not None,
                    self.column_name is not None,
                    self.Encoder is not None])  
    def getCategorizedDf(self,df:pd.DataFrame,column_name:list|str,encoding_scheme:str|int):
        #returns pipeline 
        self.dataset=df
        self.encoding_scheme=encoding_scheme
        self.column_name=column_name
        assert(column_name in get_categorical_columns(df) if isinstance(column_name,str) else all([i in get_categorical_columns(df) for i in column_name]))
        categorical_df=df.loc[:,column_name]
        if(encoding_scheme=="one_hot" or encoding_scheme==1):
            cat_pipe = make_pipeline(SimpleImputer(strategy='constant', fill_value='N/A'),
                                     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        else:
            cat_pipe=make_pipeline(
                cat_pipe = make_pipeline(SimpleImputer(strategy='constant', fill_value='N/A'),
                                     LabelEncoder())
            )
        full_pipe = ColumnTransformer([('cat', cat_pipe,self.column_name)])
         
        return full_pipe

df = pd.read_csv("datasets\Linear Regression\housing.csv")
target_column="median_house_value"

cat_columns=get_categorical_columns(df)
if target_column in cat_columns:
    cat_columns.remove(target_column)
columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_df=df[cat_columns]
df = df[columns]
df_features = df.drop(target_column, axis=1)
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

features_list=['longitude', 'housing_median_age']
target=target_column

combo2=features_list+cat_columns
df_concat=pd.concat([df,cat_df],axis=1)
X_train, X_test, y_train, y_test = train_test_split(df_concat[combo2], df_concat[target], test_size=0.2, random_state=42)
cat_pipe=Categorical().getCategorizedDf(X_train,cat_columns,1)
model = Pipeline([("cat",cat_pipe),("lr",LinearRegression())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(combo2)
print(r2)

     