# K-Nearest-Neighbour-Classifier

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks. In the context of classification, it's often referred to as the K-Nearest Neighbors Classifier. The main idea behind KNN is to predict the class of a new data point based on the majority class of its "K" nearest neighbors in the feature space.


## Implementation

- Importing the required Libraries and reading the dataset
- Preforming EDA on the data
- Standardization of the data
- Splitting the data into the Train and the Test data
- Grid Search for Tuning the Algorithm
- Building the K-Nearest Neighbor Model
- Finding the accuracy
  
## Packages Used

- pandas
- numpy
- matplotlib.pyplot
- seaborn
- warnings
- from numpy import set_printoptions
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.model_selection import GridSearchCV
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.metrics import accuracy_score, confusion_matrix, cross_val_score
