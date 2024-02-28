
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import CategoricalNB, GaussianNB

features, classification = load_breast_cancer(return_X_y=True)

cnb = CategoricalNB()
gnb = GaussianNB()

gnb.fit(features, classification)
results = gnb.predict(features)

print(list(zip(results, classification)))

scores = cross_val_score(gnb, features,classification, cv=5, scoring='f1')
print(scores)


