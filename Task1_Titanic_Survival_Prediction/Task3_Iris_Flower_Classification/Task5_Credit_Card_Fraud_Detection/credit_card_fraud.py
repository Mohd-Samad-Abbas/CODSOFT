# ============================================================
# CODSOFT DATA SCIENCE INTERNSHIP
# Task 5: Credit Card Fraud Detection
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("creditcard.csv")
print("Shape:", df.shape)
print(df['Class'].value_counts())
print(f"Fraud %: {df['Class'].value_counts(normalize=True)[1]*100:.4f}%")

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
counts = df['Class'].value_counts()
axes[0].bar(['Genuine','Fraud'], counts.values, color=['steelblue','crimson'])
axes[0].set_title("Class Distribution")
df[df['Class']==0]['Amount'].plot(kind='hist', bins=50, alpha=0.6, color='steelblue', label='Genuine', ax=axes[1])
df[df['Class']==1]['Amount'].plot(kind='hist', bins=50, alpha=0.8, color='crimson', label='Fraud', ax=axes[1])
axes[1].set_title("Amount Distribution")
axes[1].legend()
df[df['Class']==0]['Time'].plot(kind='hist', bins=50, alpha=0.6, color='steelblue', label='Genuine', ax=axes[2])
df[df['Class']==1]['Time'].plot(kind='hist', bins=50, alpha=0.8, color='crimson', label='Fraud', ax=axes[2])
axes[2].set_title("Time Distribution")
axes[2].legend()
plt.tight_layout()
plt.savefig("fraud_eda.png", dpi=100)
plt.close()

data = df.copy()
data['Amount'] = StandardScaler().fit_transform(data[['Amount']])
data['Time'] = StandardScaler().fit_transform(data[['Time']])

X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

fraud_train = X_train[y_train == 1]
genuine_train = X_train[y_train == 0]
y_fraud = y_train[y_train == 1]
y_genuine = y_train[y_train == 0]
n_fraud = len(fraud_train)
genuine_sample = genuine_train.sample(n=min(n_fraud * 10, len(genuine_train)), random_state=42)
y_genuine_sample = y_genuine.loc[genuine_sample.index]
X_train_bal = pd.concat([fraud_train, genuine_sample])
y_train_bal = pd.concat([y_fraud, y_genuine_sample])
print(f"Balanced - Fraud: {y_train_bal.sum()} | Genuine: {(y_train_bal==0).sum()}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "model": model, "y_pred": y_pred}
    print(f"{name}: {acc*100:.4f}%")

best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
y_pred_best = results[best_name]['y_pred']
print(f"\nBest Model: {best_name} ({results[best_name]['accuracy']*100:.4f}%)")
print(classification_report(y_test, y_pred_best, target_names=['Genuine','Fraud']))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Genuine','Fraud'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Reds', colorbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("fraud_confusion_matrix.png", dpi=100)
plt.close()

plt.figure(figsize=(7, 6))
for name, res in results.items():
    if hasattr(res['model'], "predict_proba"):
        y_prob = res['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.4f})")
plt.plot([0,1],[0,1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("fraud_roc_curves.png", dpi=100)
plt.close()

rf = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)
plt.figure(figsize=(9, 6))
feat_imp.sort_values().plot(kind='barh', color='crimson')
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("fraud_feature_importance.png", dpi=100)
plt.close()

print("\nSample Predictions:")
genuine_sample = X_test[y_test == 0].iloc[0:1]
fraud_sample = X_test[y_test == 1].iloc[0:1]
for label, sample in [("Genuine", genuine_sample), ("Fraud", fraud_sample)]:
    pred = best_model.predict(sample)[0]
    prob = best_model.predict_proba(sample)[0][1]
    print(f"{label} => {'FRAUD' if pred==1 else 'GENUINE'} (prob: {prob:.4f})")

print("\n[DONE] Credit Card Fraud Detection complete.")
