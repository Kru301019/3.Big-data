import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
#you can also update your sklearn lib to the latest version to use
#plot_confusion_matrix
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
from sklearn.model_selection import train_test_split

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
X_train, X_test, y_train, y_true = train_test_split(X, y)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
y_predict = lda.predict(X_test)

confusionM(y_true,y_predict,target_names)