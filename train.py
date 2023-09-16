import bentoml

from sklearn import svm, datasets


# Load training datasets
iris = datasets.load_iris()
x, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma="scale")
clf.fit(x, y)


# Save Model to BentoML local model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")
