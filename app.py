# --------------------------------------------
# HYBRID FARM AI – MODEL A:
# ADVANCED CROP RECOMMENDATION
# 4 SUPERVISED + 2 UNSUPERVISED
# --------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # NEW

from sklearn.cluster import KMeans                   # UNSUPERVISED 1
from sklearn.decomposition import PCA                # UNSUPERVISED 2

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------


# 1. LOAD DATA
path = r"E:\AIML ASSIGNMENT\Crop_recommendation.csv"   # SAME as website
data = pd.read_csv(path)


print("Dataset loaded successfully")
print(data.head())
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())

# --------------------------------------------
# 2. BASIC CHECKS
# --------------------------------------------

print("\nMissing values:")
print(data.isnull().sum())

print("\nCrop distribution:")
print(data['label'].value_counts())

# --------------------------------------------
# 3. FEATURES AND TARGET
# --------------------------------------------

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = data[feature_cols]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# --------------------------------------------
# 4. DEFINE SUPERVISED MODELS (4 Pipelines)
# --------------------------------------------

pipelines = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, multi_class="multinomial"))
    ]),

    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ]),

    "RandomForest": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(random_state=42))
    ]),

    "KNN": Pipeline([  # NEW supervised model
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])
}

# --------------------------------------------
# 5. CROSS-VALIDATION TO COMPARE MODELS
# --------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print("\nModel:", name)
    print("Cross-validation scores:", scores)
    print("Mean accuracy: {:.2f}%".format(scores.mean() * 100))

# --------------------------------------------
# 6. SELECT BEST BASE MODEL
# --------------------------------------------

best_base_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print("\nBest base model:", best_base_model_name)

# --------------------------------------------
# 7. HYPERPARAMETER TUNING FOR BEST MODEL
# --------------------------------------------

if best_base_model_name == "RandomForest":
    pipe = pipelines["RandomForest"]
    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None]
    }

elif best_base_model_name == "SVM_RBF":
    pipe = pipelines["SVM_RBF"]
    param_distributions = {
        "clf__C": [0.1, 1, 10, 50, 100],
        "clf__gamma": ["scale", 0.01, 0.001, 0.0001]
    }

elif best_base_model_name == "LogisticRegression":
    pipe = pipelines["LogisticRegression"]
    param_distributions = {
        "clf__C": [0.01, 0.1, 1, 10, 100]
    }

else:  # KNN
    pipe = pipelines["KNN"]
    param_distributions = {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]  # 1 = Manhattan, 2 = Euclidean
    }

print("\nStarting hyperparameter tuning...")
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="accuracy",
    n_jobs=-1,
    cv=cv,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(search.best_params_)
print("Best CV score: {:.2f}%".format(search.best_score_ * 100))

best_model = search.best_estimator_

# --------------------------------------------
# 8. EVALUATE ON TEST SET (BEST SUPERVISED MODEL)
# --------------------------------------------

y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print("\nFinal Test Accuracy: {:.2f}%".format(test_acc * 100))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.title("Confusion Matrix - Best Crop Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 9. UNSUPERVISED ANALYSIS (2 ALGORITHMS)
#    1) K-Means Clustering
#    2) PCA (for 2D visualization)
# --------------------------------------------

print("\n--- UNSUPERVISED LEARNING SECTION ---")

# Scale features for unsupervised learning
scaler_unsup = StandardScaler()
X_scaled = scaler_unsup.fit_transform(X)

# 9.1 K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
data["cluster_kmeans"] = clusters

print("\nK-Means cluster counts:")
print(data["cluster_kmeans"].value_counts())

# Optional: Cluster vs crop frequency
cluster_crop_counts = pd.crosstab(data["cluster_kmeans"], data["label"])
print("\nK-Means clusters vs crop labels:")
print(cluster_crop_counts)

# 9.2 PCA for Visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.6)
plt.title("PCA (2D) of Crop Dataset with K-Means Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 10. SAVE BEST SUPERVISED MODEL FOR WEBSITE
# --------------------------------------------

joblib.dump(best_model, "hybrid_crop_reco_model.pkl")
print("Model saved as hybrid_crop_reco_model.pkl")

# --------------------------------------------
# 11. DEMO PREDICTION
# --------------------------------------------

sample = np.array([[90, 42, 43, 20.0, 80.0, 6.5, 200.0]])
sample_df = pd.DataFrame(sample, columns=feature_cols)

prediction = best_model.predict(sample_df)[0]
print("\nRecommended crop for sample values:", prediction)
