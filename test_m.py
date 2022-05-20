import pandas as pd
from model_train import TrainModel
from data_compose import data_to_pandas, model_classifiers

# Loading the data into dataframe
df= data_to_pandas('train.csv')
#print(df)
#df_test=pd.read_csv('test.csv', index_col='id')
df= df.drop(columns=['f_03', 'f_04', 'f_07', 'f_16', 'f_17', 'f_27'])


X, y=df.drop('target', axis=1), df['target']

num_var=X.select_dtypes(exclude=[object]).columns.values.tolist()
#cat_var=X.select_dtypes(include=[object]).columns.values.tolist()

classifiers=model_classifiers(num_var)

models=TrainModel(classifiers)

train_results=models.train_pred(X, y)

print(train_results)
