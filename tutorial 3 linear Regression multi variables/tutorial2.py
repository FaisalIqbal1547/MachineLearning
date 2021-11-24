import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math 
df = pd.read_csv("homeprices.csv")
print(df)

median_of_bedrooms= df.bedrooms.median()
#print(median_of_bedrooms)

median_bedrooms_round_off= math.floor(df.bedrooms.median())
#print(median_bedrooms_round_off)

df.bedrooms = df.bedrooms.fillna(median_bedrooms_round_off)
#print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

#print(reg.coef_)
#print(reg.intercept_)
myprediction1=reg.predict([[3000,3,15]])
print(myprediction1)


