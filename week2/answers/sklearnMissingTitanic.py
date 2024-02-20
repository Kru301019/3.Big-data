import pandas as pd
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('N:/train.csv')
test = pd.read_csv('N:/test.csv')

train['Sex'] = train['Sex'].map( {'male':1,'female':0} )
train['Sex'] = train['Sex'].astype(int)
train['Embarked'] = train['Embarked'].map( {'Q':2,'S':1,'C':0} )
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].median())
train['Embarked'] = train['Embarked'].astype(int)

train = train.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)
trainAge = train[train['Age'].notnull()]
testAge = train[train['Age'].isnull()]
testAge.info()
print(testAge.head())

trainAgeX = trainAge.drop(['Age'],axis=1)
trainAgeY = trainAge['Age']
testAgeX = testAge.drop(['Age'],axis=1)

#random forest regressor
rf_reg = RandomForestRegressor(max_depth=5, random_state=1)
rf_reg.fit(trainAgeX, trainAgeY)
testAge.loc[:,'Age'] = rf_reg.predict(testAgeX)
testAge['Age'].hist(bins=6)
testAge.info()