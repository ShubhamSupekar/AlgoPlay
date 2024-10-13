import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def get_categorical_columns(df:pd.DataFrame, threshold:int=5):
    columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns=[]
    for i in range(len(columns)):
        if len(df.loc[:,columns[i]].unique())<=threshold:
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


