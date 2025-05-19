import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# --------------------
# 1. Load dataset
# --------------------

df = pd.read_excel("FEATURES_CAC40_features_dataset.xlsx")

features = [
    "Close Price", "Volume", "Market Cap Proxy",
    "Return_1m", "Price_Momentum", "Volume_Momentum",
    "Volatility_1m", "Average_Volume_1m", 
    "Fed Funds Rate", "ECB Interest Rate",
    "Index_CAC40", "Index_FTSEMIB", "Index_SP500", "Index_FTSE100",
    "Index_SMI", "Index_HSCEI", "Index_Nikkei225", "Index_Russell2000",
    "Index_MSCI_World", "Index_MSCI_Emerging", "Index_DowJones", "Index_DAX", 
    "Commodity_Gold", "Commodity_Silver", "Commodity_NaturalGas"
    ] # we add "6-Month Momentum" and "1-Year Momentum" for MSCI USA Momentum

X = df[features]
y = df["Label"]

X.replace([np.inf, -np.inf], np.nan, inplace=True)

# --------------------
# 2. Train-test split
# --------------------

train_idx = df["Rebalance Date"] < "2023-01-01"
test_idx = df["Rebalance Date"] >= "2023-01-01"

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------
# 3. Train models and analyze
# --------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(), random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

feature_importance_dict = {}
results = {}

plt.figure(figsize=(12, 8))

for i, (name, model) in enumerate(models.items()):
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    # Get probabilities
    y_scores = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)

    # AUCPR score
    aucpr = average_precision_score(y_test, y_scores)
    results[name] = aucpr
    print(f"{name} AUCPR: {aucpr:.4f}")

    # Feature importance
    if name in ["XGBoost", "Random Forest"]:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by="Importance", ascending=False)
        feature_importance_dict[name] = importance_df
        print(f"\nTop 10 features for {name}:")
        print(importance_df.head(10))

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=f"{name} (AUCPR={aucpr:.2f})")

    # Confusion Matrix
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Final PR Curve Plot
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
