# Movierating
MOVIE RATING PREDICTION WITH PYTHON
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




iris_data=pd.read_csv('/content/IRIS.csv')

print(iris_data.head())

print(iris_data.describe())

print(iris_data.isnull().sum())

sns.FacetGrid(iris_data,hue="species").map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris_data,hue="species").map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris_data,hue="species").map(sns.distplot,"sepal_length").add_legend()
plt.show()

sns.boxplot(x="species",y="petal_length",data=iris_data)
plt.show()

sns.violinplot(x="species",y="petal_length",data=iris_data)
plt.show()

sns.set_style("whitegrid")
sns.pairplot(iris_data,hue="species",size=3);
plt.show()

X = iris_data.drop("species", axis=1)
y = iris_data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

iris_outcome = pd.crosstab(index=iris_data["species"],  # Make a crosstab
                              columns="count")      # Name the count column

iris_outcome

iris_setosa=iris_data.loc[iris_data["species"]=="Iris-setosa"]
iris_virginica=iris_data.loc[iris_data["species"]=="Iris-virginica"]
iris_versicolor=iris_data.loc[iris_data["species"]=="Iris-versicolor"]

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)


