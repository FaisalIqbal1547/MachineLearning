import pandas as pd
import numpy as np
df = pd.read_csv("homeprices.csv")
print(df)

dumies=pd.get_dummies(df.town)
print(dumies)

merged = pd.concat([df,dumies],axis='columns')
print(merged)

final = merged.drop(['town','west windsor'],axis='columns') # according to linear regresion we have  to drop one of the column lets suppose we have five column data then we drop 1 and we have 4 similarly here we have 3 column we will drop 1 to get the 2 column (it is the rule for linear regression model)
print(final) 
# Now lets create the linear regression model 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X= final.drop('price',axis='columns')
print(X)

Y=final.price
print(Y)

model.fit(X,Y) # here we are actually training the model
# we have a very less data thats why it will take less time to train on this data
#-------------------------------------------------------------------
# now lets see how my X looks
# Now for my robinsville my prediction of price is 
#=>model.predict([[2800,0,1]])
print(model.predict([[2800,0,1]])) # here robinsville is 1 and for township we have 0
#+++++++++++++++++++++++++++++++++++++++++++++++
#Now for my township my prediction of price is 
#=>model.predict([[2800,1,0]])
print(model.predict([[2800,1,0]])) # here robinsville is 0 and for township we have 1
#+++++++++++++++++++++++++++++++++++++++++++++++
#Now for my west windsor my prediction of price is 
#=>model.predict([[2800,0,0]])
print(model.predict([[2800,0,0]])) # here robinsville is 0 and for township we have 0
#+++++++++++++++++++++++++++++++++++++++++++++++
# now I have to check the score of my model how much it is accurate 
#model.score(X,Y)
print(model.score(X,Y)) # here i am getting the score of 0.95 so it means my trained model is 95 percent accurate
#+++++++++++++++++++++++++++++++++++++++++++++++
#for using the hot encoding first you need to use the label encoding
#-------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# so df label encoding
dfle = df
dfle.town = le.fit_transform(dfle.town)
#print(dfle)
#+++++++++++++++++++++++++++++++++++++++++++++++
#X is training data set and Y is our output
X = dfle[['town','area']].values
print("dfle[['town','area']].values=>",X)
Y=dfle.price
print("dfle.price=>",Y)
#-------------------------------------------------------------------
#now lets import onehotencoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#ohe =  ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)

X1 = transformer.fit_transform(X)
X=np.array(X1)
#X = X.astype('float64')
print("X=>",X1)
X=X[:,1:]
print(X)

model.fit(X,Y)

model.predict([[1,0,2800]])

print(model.predict([[1,0,2800]]))

# so we have seen two methods for getting the dummies variable first one is pandas dummies method and second one is sklearn preprocessing
