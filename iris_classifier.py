# 🛠️ Step 1: Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 🪄 Step 2: Load and Prepare Dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)  
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# 📊 Step 3: Data Overview
print("Dataset preview:\n", iris_df.head())
print("\nBasic Info:\n", iris_df.describe())

# 📈 Step 4: Visualization
sns.heatmap(iris_df.drop('species', axis=1).corr(), annot=True, cmap='Blues')
plt.title("Feature Correlation in Iris Dataset")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# 🧪 Step 5: Train/Test Split
features = iris_df.drop('species', axis=1)  
labels = iris_df['species']                 

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

# 🧠 Step 6: Train Model
classifier = LogisticRegression(max_iter=200)
classifier.fit(X_train, y_train)

# 📊 Step 7: Evaluate
predictions = classifier.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_matrix.png")  
plt.show()
