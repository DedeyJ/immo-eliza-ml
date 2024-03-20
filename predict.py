import pickle
import pandas as pd

from train import Preprocessor

file_name = ".\data\predict.csv"
df = pd.read_csv(file_name)
df_copy = df.copy()
with open('.\model\preprocess.pkl', 'rb') as f:
    preprocess = pickle.load(f)

X = preprocess.transform(df)

with open('.\model\model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X)
df_predicted = pd.DataFrame({'Predicted': y_pred})
df = pd.concat([df_copy,df_predicted],axis=1)

df.to_csv('prediction.csv', index=False)