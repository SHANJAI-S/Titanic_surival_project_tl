import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score





# Load Titanic dataset
titanic_data = pd.read_csv(r'C:\Users\sshan\Downloads\train.csv')

# Display first few rows of the dataset
'''print(titanic_data.head())'''

# Display dataset shape
'''print("Dataset shape:", titanic_data.shape)'''

# Drop unnecessary columns
titanic_data = titanic_data.drop(columns=['Cabin'], axis=1, errors='ignore')

# Fill missing values
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# Display missing values count
'''print("Missing values count:")'''
'''print(titanic_data.isnull().sum())'''

# Encode categorical values
titanic_data = titanic_data.replace(
    {'Sex': {'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}).infer_objects(copy=False)

# Display first few rows after encoding
'''print(titanic_data.head())'''

# Feature selection
x = titanic_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket'])
y = titanic_data['Survived']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Display shape of training and test sets
'''print("Training set shape:", x_train.shape, y_train.shape)
print("Test set shape:", x_test.shape, y_test.shape)'''

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Implement Logistic Regression from scratch
class Logistic_Regression:
    def __init__(self, learning_rate=0.01, no_of_iterations=5000):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1 / self.m) * np.sum(Y_hat - self.Y)
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        return np.where(Y_pred > 0.5, 1, 0)


# Train model
model = Logistic_Regression(learning_rate=0.01, no_of_iterations=5000)
model.fit(x_train, y_train)

# Predictions & Accuracy
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Training Accuracy:", training_data_accuracy)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Test Accuracy:", test_data_accuracy)
# Compute precision, recall, and F1-score
precision = precision_score(y_test, x_test_prediction)
recall = recall_score(y_test, x_test_prediction)
f1 = f1_score(y_test, x_test_prediction)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

