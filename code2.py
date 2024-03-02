#SYED ABRAR AIE22147 ML-ASSIGNMENT 5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
from sklearn.exceptions import UndefinedMetricWarning

# Ignore warnings about ill-defined precision and F-score
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



file_path = r"D:\SA C\sem 4\MACHINE LEARING\resume_data.csv"


df = pd.read_csv(file_path)


print("Column Names:", df.columns)

required_columns = ['Years Of Experience', 'Age', 'Category']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the DataFrame.")


X = df[['Years Of Experience', 'Age']]
y = df['Category']


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

# Training the classifier
knn.fit(X_train, y_train)

# Making prediction on the test set
y_test_pred = knn.predict(X_test)

# Confusion matrix using scikit-learn's default confusion_matrix
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Plotting the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".0f")
plt.title('Confusion Matrix - Test Set')
plt.show()

# Classification report
classification_report_test = classification_report(y_test, y_test_pred)
print("Classification Report - Test Set:")
print(classification_report_test)


# Calculate MSE, RMSE, MAPE, and R2 scores


regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
# Using squared=False for RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)  
mape = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying the following results
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Percentage Error:", mape)
print("R2 Score:", r2)

#############################################################
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# A3: Generate 20 data points (training set data) consisting of 2 features (X & Y)
# whose values vary randomly between 1 & 10. Based on the values, assigning these 20 points according
# to 2 different classes (class0 - Blue & class1 – Red).

np.random.seed(42)  # For reproducibility
X_train_gen = np.random.uniform(low=1, high=10, size=(20, 2))
y_train_gen = np.random.choice([0, 1], size=20)

# Scatter plot of the training data
plt.scatter(X_train_gen[:, 0], X_train_gen[:, 1], c=y_train_gen, cmap=plt.cm.brg)
plt.title('Scatter Plot of Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# A4: Generate test set data with values of X & Y varying between 0 and 10 with increments of 0.1.
# This creates a test set of about 10,000 points. Performing classification on these points with the above training data
# using kNN classifier and setting the k value to (k = 3).

X_test_gen = np.array(np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))).T.reshape(-1, 2)

knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_gen, y_train_gen)
y_test_pred_gen = knn_classifier.predict(X_test_gen)

# A5: Making a scatter plot of the test data output with test points colored as per their predicted class colors.
# All points predicted class 0 are labeled blue color and class 1 are labeled red color accordingly.

plt.scatter(X_test_gen[:, 0], X_test_gen[:, 1], c=y_test_pred_gen, cmap=plt.cm.brg, alpha=0.5)
plt.title('Scatter Plot of Test Data Output')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# A6: Repeat the exercises A3 to A5 for your project data considering any two features and classes.
# Assuming your dataset is loaded into a DataFrame named 'df_project'
# Extracting features and target variable

# Scatter plot of the test data output

classes = np.unique(y_test_pred)
colors = plt.cm.brg(np.linspace(0, 1, len(classes)))

for i, class_label in enumerate(classes):
    class_indices = (y_test_pred == class_label)
    
    if isinstance(X_test, pd.DataFrame):
        plt.scatter(X_test.loc[class_indices, 'Years Of Experience'], 
                    X_test.loc[class_indices, 'Age'], 
                    color=colors[i], 
                    label=f'Class {class_label}', 
                    alpha=0.5)
    elif isinstance(X_test, np.ndarray):
                    X_test[class_indices, 1], 
                    color=colors[i], 
                    label=f'Class {class_label}', 
                    alpha=0.5

plt.title('Scatter Plot of Test Data Output for Project Data')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()
plt.show()
######################################################
#Q7
from sklearn.model_selection import GridSearchCV


# Extracting features and target variable
X_project = df[['Years Of Experience', 'Age']]
y_project = df['Category']

# Splitting the dataset into training and testing sets
X_train_project, X_test_project, y_train_project, y_test_project = train_test_split(X_project, y_project, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}

# Create the kNN classifier
knn = KNeighborsClassifier()

# Perform GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_project, y_train_project)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)
#########################################################
#Q5


file_path = r"D:\SA C\sem 4\MACHINE LEARING\resume_data.csv"


df = pd.read_csv(file_path)


X_project = df[['Years Of Experience', 'Age']]
y_project = df['Category']

# Convert categorical labels to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y_project_encoded = label_encoder.fit_transform(y_project)


X_train_project, X_test_project, y_train_project, y_test_project = train_test_split(X_project, y_project_encoded, test_size=0.2, random_state=42)

# Defining a range of k values to experiment with
k_values = [1, 3, 5, 7, 9]

# Plotting decision boundaries for different k values
# Decision boundary visualization
k_values = [1, 3, 5, 7, 9]

plt.figure(figsize=(15, 10))

for i, k in enumerate(k_values, 1):
    plt.subplot(2, 3, i)

    knn_project = KNeighborsClassifier(n_neighbors=k)
    knn_project.fit(X_train_project, y_train_project)

    h = .02
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # Extracting column names
    feature_names = X_train_project.columns.tolist()

    x_min, x_max = X_train_project.iloc[:, 0].min() - 1, X_train_project.iloc[:, 0].max() + 1
    y_min, y_max = X_train_project.iloc[:, 1].min() - 1, X_train_project.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn_project.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X_train_project.iloc[:, 0], X_train_project.iloc[:, 1], c=y_train_project, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'k = {k}')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

plt.show()