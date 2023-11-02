#CSC 767 HW 1 part 2
# Team Members:


# Load libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
import numpy
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

# Load dataset
url = 'winequality-white.csv'
names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
         'quality']
dataset = read_csv(url, sep=';')


# Summarize Data

# Descriptive statistics
# shape
print(dataset.shape)

# List all data types used by the DataFrame to characterize each attribute using dtypes property.

set_option('display.max_rows', 500)
print(dataset.dtypes)
# View first 20 rows
set_option('display.width', 100)
print(dataset.head(20))
# descriptions, change precision to 3 places
set_option('precision', 3)

# The describe() function list 8 statistical properties of each attribute.
print(dataset.describe())
# Group class distribution
print(dataset.groupby('quality').size())
# Correlations between attributes using Pearson's Correlation Coefficient
# 6a. Pairwise Pearson correlations
print(dataset.corr(method = 'pearson'))
# 6b.Skew of Univariate Distributions
# Knowing an attribute has a skew may allow us to perform data preparation to correct the skew and later
# improve the accuracy of our models.

print(dataset.skew())
# 6c. Visualization data with Univariate Plot
# Histograms group data into bin and provide us a count of the number of observations in each bin. 
print(dataset.hist())
pyplot.figsize = (8,8)
pyplot.savefig('histograms.png', dpi=300)
pyplot.show()

# Density plots, this help us getting a quick idea of the distribution of each attribute. 
# As we can see the distribution for each attribute is clearer than the histograms

dataset.plot(kind='density', subplots=True, layout=(4,4),sharex=False, figsize = (14,14))
pyplot.show()

# Box and Whisker Plots, this give an idea of the spread of data and dot outside of the Whisker show 
# candidate outliner values.
dataset.plot(kind='box', subplots=True, layout=(4,4),sharex=False, figsize = (14,14))
pyplot.show()

# 6d. Correlation Matrix Plot
# This gives an indication of how related the changes are between two variables.
# Plot correlation matrix
fig = pyplot.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# Correlation Matrix Generic Plot
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()

# Scatterplot Maxtrix, this shows the relationship between variables
scatter_matrix(dataset)
pyplot.figure(figsize=(20,18))

pyplot.show()

# Listing 7
# 7a. Rescaling data
# After rescalling we can see that all of the values are in the range between 0 and 1.
array = dataset.values
# seperate array into input and output components
X = array[:, 0:11]
Y = array[:,11]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5, :])

# 7b. Standardize Data
X = array[:, 0:11]
Y = array[:,11]
scaler_standard = StandardScaler().fit(X)
rescaled_standardX = scaler_standard.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaled_standardX[0:5, :])

# 7c. Normalize Data

X = array[:, 0:11]
Y = array[:,11]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5, :])

# 7d. Binarize Data
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:5, :])

