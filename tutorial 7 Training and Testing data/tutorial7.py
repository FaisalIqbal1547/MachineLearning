import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("carprices.csv")
df.head()
#print(df.head())

plt.scatter(df['Mileage'],df['Sell Price($)'])

X= df[['Mileage','Age(yrs)']]
y=df['Sell Price($)']

print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print("X_train=>",len(X_train))


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,y_train)

print(clf.predict(X_test))

print(clf.score(X_test,y_test))

