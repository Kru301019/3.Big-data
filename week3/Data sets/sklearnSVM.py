import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

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


iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

digits = datasets.load_digits()
digitsx = digits.data
X = digits.data
y = digits.target
target_names = digits.target_names


X_train, X_test, y_train, y_true = train_test_split(X, y)
#svc = svm.SVC(decision_function_shape='ovo')
#svc = svm.SVC(decision_function_shape='ovr')
#adopting the 1 vs 1 or 1 vs rest classification strategy
svc = svm.SVC(decision_function_shape='ovo',kernel='linear')
#having different kernels
#svc = svm.SVC(decision_function_shape='ovo',kernel='rbf',C=100,gamma=100)
#svc = svm.SVC(decision_function_shape='ovo',kernel='rbf',C=100,gamma=0.01)
#svc = svm.SVC(decision_function_shape='ovo',kernel='rbf',C=1,gamma=0.01)
#svc = svm.SVC(decision_function_shape='ovo',kernel='rbf',C=1,gamma=0.001)
#Penalty parameter C of the error term. it  controls the tradeoff between 
#smooth decision boundary and classifying the training points correctly.
#Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Higher the value of gamma
#more the model will try to exact fit the training data set

svc.fit(X_train,y_train)
print(svc.score(X_test,y_true))
#accuracy
y_predict = svc.predict(X_test)
print(y_predict)

confusionM(y_true,y_predict,target_names)