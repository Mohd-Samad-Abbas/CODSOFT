# ============================================================
# CODSOFT DATA SCIENCE INTERNSHIP
# Task 3: Iris Flower Classification
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("IRIS.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df['species'].value_counts())
print(df.isnull().sum())

sns.set(style="whitegrid")
pair = sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde')
pair.fig.suptitle("Iris Feature Pairplot", y=1.02)
plt.savefig("iris_pairplot.png", bbox_inches='tight', dpi=100)
plt.close()

plt.figure(figsize=(7, 5))
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_heatmap.png", dpi=100)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['sepal_length','sepal_width','petal_length','petal_width']
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=df, x='species', y=feat, palette='Set2', ax=ax)
    ax.set_title(feat.replace('_',' ').title())
plt.suptitle("Feature Distribution by Species")
plt.tight_layout()
plt.savefig("iris_boxplots.png", dpi=100)
plt.close()

X = df.drop('species', axis=1)
y = df['species']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', C=1.0, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "model": model}
    print(f"{name}: {acc*100:.2f}%")

best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
y_pred_best = best_model.predict(X_test_sc)
print(f"\nBest Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=100)
plt.close()

plt.figure(figsize=(8, 5))
names = list(results.keys())
accs = [results[n]['accuracy'] * 100 for n in names]
colors = ['#2ecc71' if n == best_name else '#74b9ff' for n in names]
bars = plt.bar(names, accs, color=colors, edgecolor='black')
plt.ylim(80, 102)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15, ha='right')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{acc:.1f}%", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=100)
plt.close()

print("\nSample Prediction:")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_sc = scaler.transform(sample)
pred = best_model.predict(sample_sc)
print(f"Input: sepal=5.1,3.5 petal=1.4,0.2 => {le.inverse_transform(pred)[0]}")
print("\n[DONE] Iris Classification complete.")
