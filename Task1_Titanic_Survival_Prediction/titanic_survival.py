# ============================================================
# CODSOFT DATA SCIENCE INTERNSHIP
# Task 1: Titanic Survival Prediction
# Description: Predict whether a Titanic passenger survived
#              using features like age, sex, ticket class, etc.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD & EXPLORE DATA
# ============================================================
print("=" * 60)
print("   TASK 1: TITANIC SURVIVAL PREDICTION")
print("=" * 60)

df = pd.read_csv("Titanic-Dataset.csv")
print("\n[1] Shape:", df.shape)
print("\n[2] First 5 Rows:\n", df.head())
print("\n[3] Statistical Summary:\n", df.describe())
print("\n[4] Missing Values:\n", df.isnull().sum())
print("\n[5] Survival Rate:\n", df['Survived'].value_counts(normalize=True).round(3))

# ============================================================
# 2. EDA VISUALIZATIONS
# ============================================================
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.countplot(data=df, x='Survived', palette='Set2', ax=axes[0,0])
axes[0,0].set_title("Survival Count")
axes[0,0].set_xticklabels(['Did Not Survive','Survived'])

sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2', ax=axes[0,1])
axes[0,1].set_title("Survival by Sex")

sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set2', ax=axes[0,2])
axes[0,2].set_title("Survival by Class")

df['Age'].dropna().plot(kind='hist', bins=30, color='steelblue', edgecolor='white', ax=axes[1,0])
axes[1,0].set_title("Age Distribution")

df['Fare'].plot(kind='hist', bins=40, color='coral', edgecolor='white', ax=axes[1,1])
axes[1,1].set_title("Fare Distribution")

sns.countplot(data=df, x='Embarked', hue='Survived', palette='Set2', ax=axes[1,2])
axes[1,2].set_title("Survival by Embarkation Port")

plt.suptitle("Titanic EDA Dashboard", fontsize=16)
plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=100)
plt.close()
print("\n[Plot Saved] titanic_eda.png")

# ============================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("   FEATURE ENGINEERING & PREPROCESSING")
print("=" * 60)

data = df.copy()
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data['Sex']      = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone']    = (data['FamilySize'] == 1).astype(int)
data['Title']      = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Title']      = data['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
data['Title']      = data['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
data['Title']      = LabelEncoder().fit_transform(data['Title'])

data.drop(columns=['PassengerId','Name','Ticket','Cabin','SibSp','Parch'], inplace=True)
print("Features:", list(data.columns))
print("Missing after prep:\n", data.isnull().sum())

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 4. MODEL TRAINING & COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("   MODEL TRAINING")
print("=" * 60)

models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":         DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, X_train_sc, y_train, cv=5)
    results[name] = {"accuracy": acc, "cv_mean": cv.mean(), "model": model}
    print(f"\n  {name}")
    print(f"    Test Accuracy : {acc*100:.2f}%")
    print(f"    CV Accuracy   : {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")

# ============================================================
# 5. BEST MODEL REPORT
# ============================================================
best_name   = max(results, key=lambda k: results[k]['accuracy'])
best_model  = results[best_name]['model']
y_pred_best = best_model.predict(X_test_sc)

print("\n" + "=" * 60)
print(f"  BEST MODEL: {best_name}  ({results[best_name]['accuracy']*100:.2f}%)")
print("=" * 60)
print(classification_report(y_test, y_pred_best, target_names=['Did Not Survive','Survived']))

cm   = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Did Not Survive','Survived'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("titanic_confusion_matrix.png", dpi=100)
plt.close()
print("[Plot Saved] titanic_confusion_matrix.png")

# ROC Curve
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {best_name}')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig("titanic_roc_curve.png", dpi=100)
    plt.close()
    print("[Plot Saved] titanic_roc_curve.png")

# Feature Importance
rf_model = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='steelblue')
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance"); plt.tight_layout()
plt.savefig("titanic_feature_importance.png", dpi=100)
plt.close()
print("[Plot Saved] titanic_feature_importance.png")

# Model Comparison
plt.figure(figsize=(8, 5))
names  = list(results.keys())
accs   = [results[n]['accuracy'] * 100 for n in names]
colors = ['#2ecc71' if n == best_name else '#74b9ff' for n in names]
bars   = plt.bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
plt.ylim(70, 95); plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison"); plt.xticks(rotation=15, ha='right')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{acc:.1f}%", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("titanic_model_comparison.png", dpi=100)
plt.close()
print("[Plot Saved] titanic_model_comparison.png")

# ============================================================
# 6. SAMPLE PREDICTION
# ============================================================
print("\n" + "=" * 60)
print("   SAMPLE PREDICTION")
print("=" * 60)
sample_df = pd.DataFrame([{
    'Pclass':1,'Sex':0,'Age':29,'Fare':100.0,
    'Embarked':0,'FamilySize':1,'IsAlone':1,'Title':2
}])
sample_sc  = scaler.transform(sample_df)
prediction = best_model.predict(sample_sc)[0]
print("  Input  : 1st class female, age 29, fare 100, alone")
print(f"  Result : {'SURVIVED' if prediction==1 else 'DID NOT SURVIVE'}")
print("\n[DONE] Titanic Survival Prediction complete.")
