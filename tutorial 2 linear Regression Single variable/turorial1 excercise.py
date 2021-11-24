import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
 
df = pd.read_csv("canada_per_capita_income.csv")
#print(df)
plt.xlabel('year',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df[["year"]], df[["per capita income (US$)"]], color='red', marker='+')

reg= linear_model.LinearRegression()
# we have basically this equation => y=mx+b here m is coefficient and b is intercept
a=reg.fit(df[["year"]],df[["per capita income (US$)"]])
#print(a)

#-------------------------------------------------------------------------------
d = pd.read_csv("years.csv")
#print(d)
p=reg.predict(d)
#print(p)

d['price'] =p

print(d)
d.to_csv("canadaPricePrediction.csv",index=False)
#-------------------------------------------------------------------------------
plt.xlabel('year',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df[["year"]], df[["per capita income (US$)"]], color='red', marker='+')
plt.plot(df[["year"]],reg.predict(df[['year']]),color='blue')
#-------------------------------------------------------------------------------