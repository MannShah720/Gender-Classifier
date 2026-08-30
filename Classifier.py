import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Mock Data [height (cm), weight (kg), shoe_size (EU)]
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

UNSEEN_DATA = [[190, 70, 43]]

# ========== Tabulate Data ==========
df = pd.DataFrame(X, columns=['height', 'weight', 'shoe_size'])
df['gender'] = Y

new_X = df[['height', 'weight', 'shoe_size']]
new_Y = df['gender']

new_unseen_data = pd.DataFrame(
    UNSEEN_DATA,
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
dt_clf = dt_clf.fit(X_train, Y_train)
dt_prediction = dt_clf.predict(new_unseen_data)
dt_score = dt_clf.score(X_test, Y_test)

# ========== Logistic Regression ==========
lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(X_train, Y_train)
lr_prediction = lr_clf.predict(new_unseen_data)
lr_score = lr_clf.score(X_test, Y_test)

# ========== K-NN ==========
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, Y_train)
knn_prediction = clf_knn.predict(new_unseen_data)
knn_score = clf_knn.score(X_test, Y_test)

# ========== Comparing the results ==========
print(f"\nPrediction for {new_unseen_data.iloc[0].tolist()}:")
print("-" * 55)

print(f"{'Model':<25} {'Prediction':<12} {'Accuracy':>10}")
print("-" * 55)

print(f"{'Decision Tree':<25} {dt_prediction[0]:<12} {dt_score * 100:>9.2f}%")
print(f"{'Logistic Regression':<25} {lr_prediction[0]:<12} {lr_score * 100:>9.2f}%")
print(f"{'K-NN':<25} {knn_prediction[0]:<12} {knn_score * 100:>9.2f}%")

print("-" * 55)