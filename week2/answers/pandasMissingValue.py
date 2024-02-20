import pandas as pd
train = pd.read_csv('N:/train.csv')
test = pd.read_csv('N:/test.csv')

print(train.isnull())
print(train.notnull())
#locate the missing values

train2 = train.dropna()
#remove records/rows with null/nan value in it
train3 = train.dropna(axis=1)
#remove columns with null/nan value in it
print(train.shape)
print(train2.shape)
print(train3.shape)

print(train.info())
train4 = train.fillna(-1)
#fill in the missing values with arguments
print(train4.info())
train5 = train.fillna(method='pad')
#fill method ‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None
train6 = train.fillna(train.mean())
train62 = train.fillna(train.median())
#use the mean or median numerical value to fill in the missing value
print(train.median())
train7 = train.fillna(train.mean()['Age':'Age'])
#be selective of the input and output
print(train7.info())