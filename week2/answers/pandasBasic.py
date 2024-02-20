import pandas as pd
import numpy as np

train = pd.read_csv('N:/train.csv')
test = pd.read_csv('N:/test.csv')

print(train.shape)
print(test.shape)
featureList = train.columns.tolist()
print(featureList)
#lsit of feature names


print(train['Survived'].head())
print(train.Survived.head())

print(train[ ['Survived','Name','Sex'] ].tail())

colnameList = []
for colname in featureList:
        if colname.endswith('d'):
            colnameList.append(colname)
        print(train[colnameList].head())
#filter with ending character of d

print(train.iloc[0])
#first record/row of data
print(train.iloc[ [0,3,100] ])
#0-th 3-th 100-th records/rows of data
print(train.iloc[5:10,4:7])
#5-10th rows and 4-7th colums of data
print(train.at[31,'Cabin'])
#element in 31-th row and column of Cabin

print(train.loc[train['Age'] >= 60, 'Age'])
#filter records with age>=60

print(train.loc[train.Embarked.isnull()])
#locate the null values

#filter and index

dfClass = pd.get_dummies(train['Pclass'],prefix='Class1Hot')
#one hot encoding for the discrete variables
#check the column name and add it to the original dataframe train

print(train['Age'].head())
train.sort_values('Age',inplace=True,ascending=False)
#based on ascending order, inplace =True to overwrite the original data
print(train['Age'].head())

train2 = train.set_index(['Fare'])
print(train2.sort_index(ascending=False).head())
#using the property of Fare as the index, it remains out of the data until you reset
train2 = train2.reset_index(['Fare'])
print(train2.sort_index(ascending=False).head())
#sorting

print(train.dtypes)
def sex2int(sex):
    if sex == 'male':
        return 1
    elif sex == 'female':
        return 0
    else:
        return -1
train['Sex'] = train['Sex'].apply(sex2int)
print(train.dtypes)

train['Embarked'] = train['Embarked'].map( {'Q':2,'S':1,'C':0} )
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].median())
train['Embarked'] = train['Embarked'].astype(int)
#map the value of embarked to integer

test = test.replace('male',1)
test = test.replace({'Sex':'female'},0)
test['Embarked'] = test['Embarked'].replace(['Q','S','C'],[2,1,0])
#replace - similar to ctrl+h

train.info()
def nan2minus(i):
    if i == np.nan:
        return -1
    else:
        return i
train3 = train.applymap(nan2minus)
train3.info()
#this one does not replace the null value in the train dataframe, please check why and fix it
#map functions on the whole dataset

train0 = train.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
print(train0.columns.tolist())
#drop data columns when axis==1

train0.to_csv('N:/Dalin.csv')
#save your dataframe