import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math 
df = pd.read_csv("homeprices.csv")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

#---------------------------------
import pickle

with open('model_pickle','wb') as f: # here you can see that i extracted and dump my train model
    pickle.dump(reg, f)
    
with open('model_pickle','rb') as f: # here i have used my saved model
    mp=pickle.load(f)
    
prediction_new_model_saved=mp.predict([[5000]]) # here you can see that i have correct prediction as i have after reg fit
print(prediction_new_model_saved)

#--------------------------------------
# now joblib library is the same as pickle to saved the data but i think its is effective when we the large numpy arrays

#from sklearn.externals import joblib
import joblib
joblib.dump(reg,'model_joblib')

mj= joblib.load('model_joblib')
joblib_output=mj.predict([[5000]])
print(joblib_output)