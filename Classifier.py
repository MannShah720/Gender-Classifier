from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Mock Data [height, weight, shoe_size]
NEW_DATA = [[190, 70, 43]]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# Decision Tree
dt_clf = tree.DecisionTreeClassifier()
dt_clf = dt_clf.fit(X, Y)
dt_prediction = dt_clf.predict(NEW_DATA)

# Logistic Regression
lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(X, Y)
lr_prediction = lr_clf.predict(NEW_DATA)

# K-NN
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X, Y)
knn_prediction = clf_knn.predict(NEW_DATA)

# Comparing the results
print(f"New data: {NEW_DATA[0]}")
print(f"Decision Tree Prediction: {dt_prediction}")
print(f"Logistic Regression Prediction: {lr_prediction}")
print(f"K-NN Prediction: {knn_prediction}")