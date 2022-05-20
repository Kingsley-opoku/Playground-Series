import csv
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
#from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
#from lightgbm              import LGBMClassifier


def data_to_pandas(pt: csv) ->pd.DataFrame:
    data=pd.read_csv(pt, index_col='id')
    return data
    


def __make_pipe(num_vars):
    num_4_treeModels = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

   # cat_object_type= Pipeline(steps=[
    #('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    #])
  
    tree_prepro = ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
    #('cat', cat_object_type, cat_vars),
    ], remainder='drop') 
    
    return tree_prepro


def model_classifiers(num_vars):
    tree_classifiers = {
   # "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Extra Trees":   ExtraTreesClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    #"AdaBoost":      AdaBoostClassifier(random_state=0),
    "Skl GBM":       GradientBoostingClassifier(random_state=0),
    #"Skl HistGBM":   HistGradientBoostingClassifier(random_state=0),
    "XGBoost":       XGBClassifier(),
    #"LightGBM":      LGBMClassifier(random_state=0),
    
    }
    tree_classifiers = {name: make_pipeline(__make_pipe(num_vars), model) for name, model in
                                     tree_classifiers.items()}
    return tree_classifiers