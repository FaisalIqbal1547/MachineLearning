import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
 
df = pd.read_csv("homeprices.csv")
print(df)
#matplotlib inline
'''
plt.xlabel('area(sqr ft')
plt.ylabel('price(US$')
plt.scatter(df.area, df.price, color='red', marker='+')
'''
reg= linear_model.LinearRegression()

# we have basically this equation => y=mx+b here m is coefficient and b is intercept
a=reg.fit(df[["area"]],df.price)
#print(a)
b=reg.predict([[3300]])
print("prediction=>",b)
print(reg.coef_) 
print(reg.intercept_)

m=reg.coef_*3300+reg.intercept_
print(m) # as you can see that this value and the value of prediction is the same
#-------------------------------------------------------------------------------
d = pd.read_csv("areas.csv")
#print(d)
d.head(3)
#print(d.head(3))

p=reg.predict(d)
#print(p)

d['price'] =p

#print(d)
#d.to_csv("prediction.csc",index=False)
#-------------------------------------------------------------------------------
plt.xlabel('area',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')