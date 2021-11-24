import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("insurance_data.csv")
print(df)

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,train_size=0.9)

#lets locate the test
#print("X_train=>")
#print(X_train)

#-----------------------------------------------------------------
print("X_test=>")
print(X_test)
#Now lets do the logistic regression (as sigmoid or logit function is same so from them we taken the logistic function)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train,y_train)

model.predict(X_test)

print(model.predict(X_test))
#-----------------------------------------------
print(model.predict_proba(X_test)) # this model is used for the prediction probability for our test dataset 1 class vs other class 
# so our first class is for customer they will not buy the insurance
#so our second class is for customer they will buy the insurance


