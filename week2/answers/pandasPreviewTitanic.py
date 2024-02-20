import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('N:/train.csv')
test = pd.read_csv('N:/test.csv')

print(train.shape)
#(891,12),891 records for training, each record has 12 items
print(test.shape)
#(418,11),418 records for testing/prediction, each record has 11 items, survival is excluded.

#12 items including
#PassengerID (no use)
#Survived: target value for prediction
#Pclass: ticket class
#Name(title included)
#Sex
#Age
#SibSp: number of siblings and spouses on ship
#Parch: number of parents and children on ship
#Ticket: ticket No.
#Fare: price of ticket
#Cabin: cabin No.
#Embarked: port of embarkation, C=Cherbourg, Q=Queenstown, S=Southampton

print(train.head(20))
#check the first 20 records to have a quick view
print(train.tail())
#check the last 5 records
print(train.dtypes)
#types of the features, object is strting and cant be processed directly

featureList = train.columns.tolist()
print(featureList)
#lsit of feature names


train.info()
#Missing value
#age, 714 non-null, 177 missing
#cabin, 204 non-null, 687 missing
#embarked 889 non-null, 2 missing
test.info()
#Missing value
#age, 332 non-null, 86 missing
#fare 417 non-null, 1 missing
#cabin, 91 non-null, 327 missing


print(train.iloc[:,2:].describe())
print(test.iloc[:,1:].describe())
#get a rough idea of the numerical summary
#ignore the passenger id and the survival

fig = plt.figure(figsize=(12,12))
sns.heatmap(train.iloc[:,1:].corr(),vmax=1,square=True,annot=True)
#correlation between current numerical items/features
#fair and pclass is more related to the survival rate at the first sight.

print(train['Survived'].value_counts())
print(train['Pclass'].value_counts())
print(train['Sex'].value_counts())
print(train['Age'].value_counts())
print(train['SibSp'].value_counts())
print(train['Parch'].value_counts())
print(train['Ticket'].value_counts())
print(train['Fare'].value_counts())
print(train['Cabin'].value_counts())
print(train['Embarked'].value_counts())
#have a quick look at the enumeration of each item/feature

print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived'))
print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived'))
print(train[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by='Survived'))
print(train[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived'))
print(train.groupby(['Sex','Survived'])['Survived'].count())
#group together and appreciate the difference using groupby function
pclassGrid = sns.FacetGrid(train, col='Survived', row='Pclass', height=4, aspect=1.5)
pclassGrid.map(plt.hist, 'Age', alpha=1, bins=20)
pclassGrid.add_legend()
#illustrate the results


fig = plt.figure()
sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=train)
fig = plt.figure()
sns.boxplot(x='Survived', y='Age', hue='Pclass', data=train)
fig = plt.figure()
sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=train)


fig = plt.figure()
sns.displot(train['SibSp'],bins=3,kde=False)
fig = plt.figure()
sns.displot(train['Parch'],bins=3,kde=False)
fig = plt.figure()
sns.displot(train['Age'],bins=20)
#an error would occur, please check the reason
plt.show()
