import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('N:/train.csv')
test = pd.read_csv('N:/test.csv')

fig = plt.figure()
sns.boxplot(x='Survived', y='Age', hue='Pclass', data=train)

train2 = train.dropna()

perAge = np.percentile(train2['Age'],[0,25,50,75,100])
IQR = perAge[3]-perAge[1]
upperAge = perAge[3]+IQR*1.5
lowerAge = perAge[1]-IQR*1.5
print(train2.loc[(train2['Age'] > upperAge) | (train2['Age'] < lowerAge), 'Age'])
#outliers identified with age>upper or age<lower
#please check the results and explain why

train3 = train2[(train2['Pclass']==3) & (train2['Survived']==0)]
perAge = np.percentile(train3['Age'],[0,25,50,75,100])
IQR = perAge[3]-perAge[1]
upperAge = perAge[3]+IQR*1.5
lowerAge = perAge[1]-IQR*1.5
print(train3.loc[(train3['Age'] > upperAge) | (train3['Age'] < lowerAge), 'Age'])
#here we use 1.5 IQR to examine the outliers