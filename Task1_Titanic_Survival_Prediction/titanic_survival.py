# ============================================================
# CODSOFT DATA SCIENCE INTERNSHIP
# Task 1: Titanic Survival Prediction
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Titanic-Dataset.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
sns.countplot(data=df, x='Survived', palette='Set2', ax=axes[0,0])
axes[0,0].set_title("Survival Count")
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2', ax=axes[0,1])
axes[0,1].set_title("Survival by Sex")
sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set2', ax=axes[0,2])
axes[0,2].set_title("Survival by Class")
df['Age'].dropna().plot(kind='hist', bins=30, color='steelblue', ax=axes[1,0])
axes[1,0].set_title("Age Distribution")
df['Fare'].plot(kind='hist', bins=40, color='coral', ax=axes[1,1])
axes[1,1].set_title("Fare Distribution")
sns.countplot(data=df, x='Embarked', hue='Survived', palette='Set2', ax=axes[1,2])
axes[1,2].set_title("Survival by Embarkation")
plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=100)
plt.close()

data = df.copy()
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
data['Title'] = data['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
data['Title'] = LabelEncoder().fit_transform(data['Title'])
data.drop(columns=['PassengerId','Name','Ticket','Cabin','SibSp','Parch'], inplace=True)

X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
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
print(classification_report(y_test, y_pred_best, target_names=['Did Not Survive','Survived']))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Did Not Survive','Survived'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("titanic_confusion_matrix.png", dpi=100)
plt.close()

if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("titanic_roc_curve.png", dpi=100)
    plt.close()

rf_model = results["Random Forest"]["model"]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='steelblue')
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("titanic_feature_importance.png", dpi=100)
plt.close()

print("\nSample Prediction:")
sample_df = pd.DataFrame([{'Pclass':1,'Sex':0,'Age':29,'Fare':100.0,'Embarked':0,'FamilySize':1,'IsAlone':1,'Title':2}])
sample_sc = scaler.transform(sample_df)
prediction = best_model.predict(sample_sc)[0]
print(f"1st class female, age 29 => {'SURVIVED' if prediction==1 else 'DID NOT SURVIVE'}")
print("\n[DONE] Titanic Prediction complete.")
