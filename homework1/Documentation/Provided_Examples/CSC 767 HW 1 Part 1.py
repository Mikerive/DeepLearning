#CSC 767 HW 1 part 1




# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
filename = 'WhiteWineQuality.csv'
names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides' ,  
         'free sulfur dioxide' , 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
dataset = read_csv(filename, names=names)

# Summarize Data

# Descriptive statistics
# shape
print('Number of Data Rows = ', dataset.shape[0]);
print('Number of Data columns = ', dataset.shape[1]);

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('quality').size())

# Data visualizations

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False, figsize=(16,12))
pyplot.show()

# histograms
dataset.hist(color='#607c8e',figsize=(14,11))
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset,figsize=(17,17));
pyplot.show()

# Prepare Data

# Split-out validation dataset
array = dataset.values
X = array[:,0:11]
Y = array[:,11]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, 
                                                                random_state=seed)

# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure(figsize=(10,7))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

import numpy as np

# Make predictions on validation dataset using K-Nearest Neighbor
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print('Prediction on Validation dataset using K-Nearest Neighbor')
print('Validation Accuracy = ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix = '); print(confusion_matrix(Y_validation, predictions));print()
print('Normalized Confusion Matrix = ')
print(np.around((confusion_matrix(Y_validation, predictions) / 
                 confusion_matrix(Y_validation, predictions).max()),5));
print();
print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset using Decision Tree Classifier 
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print('Prediction on Validation dataset using Decision Tree Classifier')
print('Validation Accuracy = ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix = '); print(confusion_matrix(Y_validation, predictions));print()
print('Normalized Confusion Matrix = ')
print(np.around((confusion_matrix(Y_validation, predictions)/confusion_matrix(Y_validation, 
                                                                              predictions).max()),5));
print();

print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset using Logistic Regression
LR = LogisticRegression()
LR.fit(X_train, Y_train)
predictions = LR.predict(X_validation)
print('Prediction on Validation dataset using Logistic Regression')
print('Validation Accuracy = ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix = '); print(confusion_matrix(Y_validation, predictions));print()
print('Normalized Confusion Matrix = ')
print(np.around((confusion_matrix(Y_validation, predictions)/confusion_matrix(Y_validation, 
                                                                              predictions).max()),5));
print();

print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset using Linear Discriminant Analysis 
LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, Y_train)
predictions = LDA.predict(X_validation)
print('Prediction on Validation dataset using Linear Discriminant Analysis')
print('Validation Accuracy = ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix = '); print(confusion_matrix(Y_validation, predictions));print()
print('Normalized Confusion Matrix = ')
print(np.around((confusion_matrix(Y_validation, predictions)/confusion_matrix(Y_validation, 
                                                                              predictions).max()),5));
print();

print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset using Gaussian NB 
NB = GaussianNB()
NB.fit(X_train, Y_train)
predictions = NB.predict(X_validation)
print('Prediction on Validation dataset using Gaussian NB()')
print('Validation Accuracy = ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix = '); print(confusion_matrix(Y_validation, predictions));print()
print('Normalized Confusion Matrix = ')
print(np.around((confusion_matrix(Y_validation, predictions)/confusion_matrix(Y_validation, 
                                                                              predictions).max()),5));
print();

print(classification_report(Y_validation, predictions))

# Make predictions on validation dataset using SVM 
SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print('Prediction on Validation dataset using SVM')
print('Validation Accuracy = ', accuracy_score(Y_validation, predictions))
print('Confusion Matrix = '); print(confusion_matrix(Y_validation, predictions));print()
print('Normalized Confusion Matrix = ')
print(np.around((confusion_matrix(Y_validation, predictions)/confusion_matrix(Y_validation, 
                                                                              predictions).max()),5));
print();

print(classification_report(Y_validation, predictions))
