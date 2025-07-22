
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

st.set_page_config(page_title="Breast Cancer Diagnostic Report", layout="wide")

# Title and Description
st.title("🩺 Breast Cancer Diagnostic Report")
st.markdown("""
This AI-based diagnostic system predicts whether a tumor is **benign** or **malignant** based on cell nucleus features.
Upload patient features or enter manually to get real-time diagnostic evaluation.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("breast_cancer_data.csv")
    data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])  # M=1, B=0
    return data

data = load_data()
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = data['diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sidebar: Custom input
st.sidebar.header("🔬 Enter Tumor Features")
input_data = {}
for feature in X.columns:
    min_val = float(data[feature].min())
    max_val = float(data[feature].max())
    mean_val = float(data[feature].mean())
    input_data[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)

# Format and scale input
user_input_df = pd.DataFrame([input_data])
user_input_scaled = scaler.transform(user_input_df)
user_pred = model.predict(user_input_scaled)
user_prob = model.predict_proba(user_input_scaled)[0]

# Prediction Result
st.subheader("🔎 Diagnostic Prediction for Custom Input")
if user_pred[0] == 1:
    st.error("⚠️ The tumor is likely **Malignant (Cancerous)**.")
else:
    st.success("✅ The tumor is likely **Benign (Non-cancerous)**.")

st.write(f"🧪 Prediction Probability → Malignant: {user_prob[1]:.2f}, Benign: {user_prob[0]:.2f}")

# Evaluation Report
st.subheader("📊 Model Evaluation Report")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    st.metric("ROC-AUC", f"{roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")

with col2:
    st.markdown("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

# Confusion matrix
st.subheader("🧾 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig1)

# Feature importance
st.subheader("📌 Feature Importance")
feat_importance = model.feature_importances_
feat_names = X.columns
imp_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_importance}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots(figsize=(8, 10))
sns.barplot(y=imp_df["Feature"], x=imp_df["Importance"], palette="viridis")
plt.title("Top Features Driving Diagnosis")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by a Machine Learning Engineer | Dataset: [UCI Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))")
