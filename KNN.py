import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("Dataset Shape:", df.shape)
print("Target Names:", iris.target_names)

def plot_features(df, x_label, y_label, title):
    plt.figure(figsize=(7,5))
    for target, color, marker, label in zip(
        [0,1,2],
        ["green","blue","red"],
        ["+","^","o"],
        iris.target_names
    ):
        subset = df[df.target == target]
        plt.scatter(subset[x_label], subset[y_label],
                    color=color, marker=marker, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

plot_features(df, "sepal length (cm)", "sepal width (cm)", "Sepal Length vs Width")
plot_features(df, "petal length (cm)", "petal width (cm)", "Petal Length vs Width")

X = df.drop(["target"], axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

train_acc = knn.score(X_train, y_train)
test_acc = knn.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Testing Accuracy: {test_acc:.2f}")

y_pred = knn.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sn.heatmap(cm, annot=True, cmap="Blues", fmt="d",
           xticklabels=iris.target_names,
           yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
