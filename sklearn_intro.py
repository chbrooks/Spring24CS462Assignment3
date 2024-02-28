
from sklearn.datasets import load_iris


from sklearn.naive_bayes import CategoricalNB

features, classification = load_iris(return_X_y=True)

cnb = CategoricalNB()
#gnb = GaussianNB()

cnb.fit(features, classification)
results = cnb.predict(features)

print(list(zip(results, classification)))



