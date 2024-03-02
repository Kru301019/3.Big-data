import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.feature import hog

def confusionM(y_true,y_predict,target_names):
#function for visualisation
    cMatrix = confusion_matrix(y_true,y_predict)
    df_cm = pd.DataFrame(cMatrix,index=target_names,columns=target_names)
    plt.figure(figsize = (6,4))
    cm = sns.heatmap(df_cm,annot=True,fmt="d")
    cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(),rotation=90)
    cm.xaxis.set_ticklabels(cm.xaxis.get_ticklabels(),rotation=0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

digits = datasets.load_digits()
digitsx = digits.data
X = digits.data
y = digits.target
target_names = digits.target_names
#model with only original data
X_train, X_test, y_train, y_true = train_test_split(X, y)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
y_predict = lda.predict(X_test)
print(lda.score(X_test,y_true))
print(y_predict)
confusionM(y_true,y_predict,target_names)

def getHoG(X):
#extract HoG features from the images
    ndarray_hog = np.empty((0,36),float)
    for row in X:
        features = hog(row.reshape((8,8)), orientations=9, pixels_per_cell=(4,4),cells_per_block=(1,1),visualize=False)
        features = features.reshape(1,36)
        ndarray_hog = np.vstack([ndarray_hog,features])
    return ndarray_hog

#model with only simple extracted features
Xf_train = getHoG(X_train)
Xf_test = getHoG(X_test)
lda_f = LinearDiscriminantAnalysis()
lda_f.fit(Xf_train,y_train)
yf_predict = lda_f.predict(Xf_test)
print(lda_f.score(Xf_test,y_true))
print(yf_predict)
confusionM(y_true,yf_predict,target_names)

#model with both original data and simple extracted features
Xf_train2 = np.hstack((X_train,Xf_train))
Xf_test2 = np.hstack((X_test,Xf_test))
lda_f2 = LinearDiscriminantAnalysis()
lda_f2.fit(Xf_train2,y_train)
yf_predict2 = lda_f2.predict(Xf_test2)
print(lda_f2.score(Xf_test2,y_true))
print(yf_predict2)
confusionM(y_true,yf_predict2,target_names)

#Check the difference of the accuracies
#See if the feature extraction helps improve the accuracy