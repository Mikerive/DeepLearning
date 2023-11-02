#CSC 767 HW 1 part 3
# Team Members:



# Load libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns

# Load dataset
url = 'winequality-white.csv'
names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
         'quality']
dataset = read_csv(url, sep=';')

# Part 3
# Listing 8
# inline plotting instead of popping out
%matplotlib inline

sns.set(style='whitegrid', context='notebook')
# Check correleation between the variables using Seaborn's pairplot. 
sns.pairplot(dataset, height=2.5)
pyplot.tight_layout()
pyplot.savefig('seaborn-scatter1.png', dpi=300)
pyplot.show()

from collections import Counter
Counter(dataset['quality'])

# Count of the target variable
sns.countplot(x='quality', data=dataset)

#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed acidity', data = dataset)

sns.boxplot('quality', 'volatile acidity', data = dataset)

corr_matrix = dataset.corr()
pyplot.figure(figsize=(11,9))
dropSelf = np.zeros_like(corr_matrix)
dropSelf[np.triu_indices_from(dropSelf)] = True

sns.heatmap(corr_matrix, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, 
            fmt=".2f", mask=dropSelf)

sns.set(font_scale=1.5)

#Correlation plot
corr_matrix = dataset.corr()
pyplot.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,linewidths=.5,center=0,cbar=False,cmap="YlGnBu")
pyplot.show()

from scipy.stats import norm 
pyplot.figure(figsize = (20,22))

for i in range(1,12):
    pyplot.subplot(5,4,i)
    sns.distplot(dataset[dataset.columns[i]], fit=norm)
	
# Quality & Alcohol Relation
pyplot.figure(figsize=(8,7))
sns.scatterplot(x='quality', 
                y='alcohol', 
                hue = 'quality',
                data=dataset);
pyplot.xlabel('Quality',size=15)
pyplot.ylabel('Alcohol', size =15)
pyplot.show()

# Quality & Volatile Acidity
f, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(14, 8))
f.suptitle('Wine Quality & Acidity', fontsize=14)
sns.violinplot(x='quality', y='fixed acidity', data=dataset, split=True, inner='quart',
               linewidth=1.3, ax=ax1)
ax1.set_xlabel("Wine Quality Class",size = 15,alpha=0.8)
ax1.set_ylabel("Wine Fixed Acidity",size = 15,alpha=0.8)

sns.violinplot(x='quality', y='volatile acidity', data=dataset, split=True, inner='quart',
               linewidth=1.3, ax=ax2)
ax2.set_xlabel("Wine Quality Class",size = 15,alpha=0.8)
ax2.set_ylabel("Wine Volatile Acidity",size = 15,alpha=0.8)
pyplot.show()

# Sulfur Dioxide Distribution in Wine Quality 
pyplot.figure(figsize=(12,8))
sns.scatterplot(x='total sulfur dioxide', y='free sulfur dioxide', hue='quality',data=dataset);
pyplot.xlabel('total sulfur dioxide',size=15)
pyplot.ylabel('free sulfur dioxide', size =15)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
             'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] 
y = dataset.quality

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

train_accuracy = lr.score(X_train, y_train)
test_accuracy = lr.score(X_test, y_test)
print('Accuracy in Train Group   : {:.2f}'.format(train_accuracy), 
      'Accuracy in Test  Group   : {:.2f}'.format(test_accuracy), sep='\n')
	  
# Confusion Matrix 
from sklearn.metrics import confusion_matrix as cm

predictions = lr.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
pyplot.xlabel('Predicted Values')
pyplot.ylabel('Actual Values')
pyplot.title('Accuracy Score: {0}'.format(score), size = 15)
pyplot.show()

# Cross Validation
X = dataset.drop(['quality'], axis=1)
y = dataset.quality
y = np.array(y)

pyplot.style.use('fivethirtyeight')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
print("Number of Rows in    Training dataset :  {} ".format(len(X_train)))
print("Number of Targets in Training dataset :  {} ".format(len(y_train)))
print("Number of Rows in    Test dataset :  {} ".format(len(X_test)))
print("Number of Targets in Test dataset :  {} ".format(len(y_test)))

sns.countplot(y_test)
pyplot.ylim((0,800))

pyplot.figure(figsize=(15,9))
y_list = [y, y_train, y_test]
titles = ['All Data','Train Data', 'Test Data']

for i in range(1,4):
    pyplot.subplot(1,3,i)
    sns.countplot(y_list[i-1])
    pyplot.title(titles[i-1])
	
#Hexbin plot
sns.jointplot(x="density", y = "residual sugar",kind = "hex", data = dataset)
sns.jointplot(x="free sulfur dioxide", y = "total sulfur dioxide",kind = "hex", data = dataset)
sns.jointplot(x="density", y = "alcohol",kind = "hex", data = dataset)

#Relationship Plots
sns.relplot(x="density", y = "residual sugar", hue = "quality", data=dataset, palette =sns.light_palette("navy",n_colors=7) )
sns.relplot(x="free sulfur dioxide", y = "total sulfur dioxide", hue = "quality", data=dataset, palette =sns.light_palette("navy",n_colors=7))

sns.relplot(x="density", y = "alcohol", hue = "quality", data=dataset, palette =sns.light_palette("navy",n_colors=7))

#Scatter: Colroed Quality
sns.set(style = "ticks")
sns_plot = sns.pairplot(dataset, hue = "quality")
sns_plot.savefig("snsScatterPlot.png")

# scatter plot matrix using seaborn (using points only)

g = sns.PairGrid(dataset)
g.map(plt.scatter);
plt.show()

# scatter plot matrix using seaborn (using points and bars)

g = sns.PairGrid(dataset)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);

# Contour plots of attributes

g = sns.PairGrid(dataset)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);

# Histogram 

sns.pairplot(dataset);

# Histograms on each attributes

f, axes = plt.subplots(4, 3, figsize=(17, 14))

sns.distplot(dataset_only[:,10], hist=True, color="b", ax=axes[0, 0]).set(title = 'Alcohol')
sns.distplot(dataset_only[:,4], hist=True, color="r", ax=axes[0, 1]).set(title = 'Chlorides')
sns.distplot(dataset_only[:,2], hist=True, color="g", ax=axes[0, 2]).set(title = 'Citric acid')
sns.distplot(dataset_only[:,7], hist=True, color="b", ax=axes[1, 0]).set(title = 'Density')
sns.distplot(dataset_only[:,0], hist=True, color="r", ax=axes[1, 1]).set(title = 'Fixed Acidity')
sns.distplot(dataset_only[:,5], hist=True, color="g", ax=axes[1, 2]).set(title = 'Free Sulfur dioxide')
sns.distplot(dataset_only[:,8], hist=True, color="b", ax=axes[2, 0]).set(title = 'pH')
sns.distplot(dataset_only[:,11], hist=True, color="r", ax=axes[2, 1]).set(title = 'quality')
sns.distplot(dataset_only[:,3], hist=True, color="g", ax=axes[2, 2]).set(title = 'Residual Sugar')
sns.distplot(dataset_only[:,9], hist=True, color="b", ax=axes[3, 0]).set(title = 'Suphates')
sns.distplot(dataset_only[:,6], hist=True, color="r", ax=axes[3, 1]).set(title = 'Total sulfur dioxide')
sns.distplot(dataset_only[:,1], hist=True, color="g", ax=axes[3, 2]).set(title = 'Violatile Acidity')

#box plot

f, axes = plt.subplots(4, 3, figsize=(19, 14))

sns.boxplot(dataset_only[:,0],color="r",ax=axes[0, 0]).set(title = 'fixed acidity')
sns.boxplot(dataset_only[:,1],color="g",ax=axes[0, 1]).set(title = 'volatile acidity')
sns.boxplot(dataset_only[:,2],color="y",ax=axes[0, 2]).set(title = 'citric acid')
sns.boxplot(dataset_only[:,3],color="r",ax=axes[1, 0]).set(title = 'residual sugar')
sns.boxplot(dataset_only[:,4],color="g",ax=axes[1, 1]).set(title = 'chlorides')
sns.boxplot(dataset_only[:,5],color="y",ax=axes[1, 2]).set(title = 'free sulfur dioxide')
sns.boxplot(dataset_only[:,6],color="r",ax=axes[2, 0]).set(title = 'total sulfur dioxide')
sns.boxplot(dataset_only[:,7],color="g",ax=axes[2, 1]).set(title = 'density')
sns.boxplot(dataset_only[:,8],color="y",ax=axes[2, 2]).set(title = 'pH')
sns.boxplot(dataset_only[:,9],color="r",ax=axes[3, 0]).set(title = 'sulphates')
sns.boxplot(dataset_only[:,10],color="g",ax=axes[3, 1]).set(title = 'alcohol')
sns.boxplot(dataset_only[:,11],color="y",ax=axes[3, 2]).set(title = 'quality')

g = sns.pairplot(dataset, diag_kind="kde")

#names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides' ,  
 #        'free sulfur dioxide' , 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

#f, axes = plt.subplots(4, 4, figsize=(19, 14))
sns.set(style="white", color_codes=True)
sns.jointplot(x=dataset["fixed acidity"], y=dataset["fixed acidity"], kind='scatter', 
              color="skyblue")
			  

