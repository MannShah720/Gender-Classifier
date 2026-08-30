import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ========== Mock Data ==========
# Features: height (cm), weight (kg), shoe_size (EU)

X = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],
    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],
    [181, 85, 43]
]

Y = [
    'male', 'male', 'female', 'female', 'male', 'male',
    'female', 'female', 'female', 'male', 'male'
]

NEW_DATA = [[190, 70, 43]]


# ========== Tabulate Data ==========

df = pd.DataFrame(
    X,
    columns=['height', 'weight', 'shoe_size']
)

df['gender'] = Y


new_X = df[['height', 'weight', 'shoe_size']]
new_Y = df['gender']


new_unseen_data = pd.DataFrame(
    NEW_DATA,
    columns=['height', 'weight', 'shoe_size']
)


# ========== Train / Test Split ==========

X_train, X_test, Y_train, Y_test = train_test_split(
    new_X,
    new_Y,
    test_size=0.2,
    random_state=42
)


# ========== Decision Tree ==========

dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)

dt_test_predictions = dt_clf.predict(X_test)
dt_score = accuracy_score(Y_test, dt_test_predictions)

dt_unseen_prediction = dt_clf.predict(new_unseen_data)


# ========== Logistic Regression ==========

lr_clf = LogisticRegression()
lr_clf.fit(X_train, Y_train)

lr_test_predictions = lr_clf.predict(X_test)
lr_score = accuracy_score(Y_test, lr_test_predictions)

lr_unseen_prediction = lr_clf.predict(new_unseen_data)


# ========== K-NN with 3 neighbors ==========

clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, Y_train)

knn_test_predictions = clf_knn.predict(X_test)
knn_score = accuracy_score(Y_test, knn_test_predictions)

knn_unseen_prediction = clf_knn.predict(new_unseen_data)

# ========== K-NN with 5 neighbors ==========

clf_knn_5 = KNeighborsClassifier(n_neighbors=5)
clf_knn_5.fit(X_train, Y_train)

knn_5_test_predictions = clf_knn_5.predict(X_test)
knn_5_score = accuracy_score(Y_test, knn_5_test_predictions)

knn_5_unseen_prediction = clf_knn_5.predict(new_unseen_data)


# ========== Comparing the Results ==========

print(f"\nPrediction for {new_unseen_data.iloc[0].tolist()}:")
print("-" * 55)

print(f"{'Model':<25} {'Prediction':<12} {'Accuracy':>10}")
print("-" * 55)

print(
    f"{'Decision Tree':<25} "
    f"{dt_unseen_prediction[0]:<12} "
    f"{dt_score * 100:>9.2f}%"
)

print(
    f"{'Logistic Regression':<25} "
    f"{lr_unseen_prediction[0]:<12} "
    f"{lr_score * 100:>9.2f}%"
)

print(
    f"{'K-NN (3 neighbors)':<25} "
    f"{knn_unseen_prediction[0]:<12} "
    f"{knn_score * 100:>9.2f}%"
)

print(
    f"{'K-NN (5 neighbors)':<25} "
     f"{knn_5_unseen_prediction[0]:<12} "
     f"{knn_5_score * 100:>9.2f}%"
)

print("-" * 55)