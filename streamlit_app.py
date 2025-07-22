
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Page config
st.set_page_config(page_title="Breast Cancer Diagnosis App", layout="wide")

# Branding
st.sidebar.image("breast_cancer.jpg", width=150)
st.sidebar.title("🩺 Breast Cancer Diagnostic Tool")

# Theme toggle
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown('<style>body { background-color: #1e1e1e; color: white; }</style>', unsafe_allow_html=True)
    plt.style.use("dark_background")
else:
    plt.style.use("default")


# Load data
data = pd.read_csv("breast_cancer_data.csv")
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
with st.spinner("Training model..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analysis", "📄 Report"])

# Prediction tab
with tab1:
    st.subheader("🔬 Enter Tumor Features")
    col1, col2 = st.columns(2)
    user_input = {}
    for i, col in enumerate(X.columns):
        with col1 if i % 2 == 0 else col2:
            user_input[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    input_df = pd.DataFrame([user_input])

    if st.button("Diagnose"):
        with st.spinner("Predicting..."):
            prediction = model.predict(input_df)[0]
            result = "🟢 Benign (Non-cancerous)" if prediction == 0 else "🔴 Malignant (Cancerous)"
            st.success(f"Prediction: {result}")
            st.info(f"Prediction Probability (Malignant): {model.predict_proba(input_df)[0][1]:.2f}")



# Analysis tab


with tab2:
    
    st.subheader("📈 Model Evaluation Metrics")
        
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.write("**ROC-AUC Score:**", roc_auc_score(y_test, proba))
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))

    
    st.subheader("📊 Visual Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.markdown("#### 🔥 Feature Importance")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig2, ax2 = plt.subplots()
        feat_imp.plot(kind='bar', ax=ax2)
        ax2.set_title("Top Feature Importances")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 📊 Diagnosis Count Plot")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='diagnosis', data=data, palette='Set2', ax=ax3)
        ax3.set_xticklabels(['Benign', 'Malignant'])
        st.pyplot(fig3)

    with col4:
        st.markdown("#### 📉 Correlation Heatmap (Top 10 Features)")
        top_feats = feat_imp.head(10).index
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(data[top_feats].corr(), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

# Report tab
with tab3:
    st.subheader("📄 Export Report as PDF")
    import io
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Breast Cancer Diagnosis Report', 0, 1, 'C')

    
    import tempfile
    import os
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    def save_plot_to_image(fig, filename):
        canvas = FigureCanvas(fig)
        canvas.draw()
        fig.savefig(filename)

    if st.button("Generate PDF Report"):
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        pdf.multi_cell(0, 10, f"ROC-AUC Score: {roc_auc_score(y_test, proba):.2f}")
        pdf.multi_cell(0, 10, f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Save plots as images
        with tempfile.TemporaryDirectory() as tmpdirname:
            cm_path = os.path.join(tmpdirname, "cm.png")
            feat_path = os.path.join(tmpdirname, "feature_importance.png")
            count_path = os.path.join(tmpdirname, "count_plot.png")
            corr_path = os.path.join(tmpdirname, "correlation.png")

            save_plot_to_image(fig, cm_path)
            save_plot_to_image(fig2, feat_path)
            save_plot_to_image(fig3, count_path)
            save_plot_to_image(fig4, corr_path)

            for img in [cm_path, feat_path, count_path, corr_path]:
                pdf.image(img, w=170)

            pdf_path = "breast_cancer_diagnosis_report.pdf"
            pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="diagnosis_report.pdf">📥 Download Full Report with Graphs</a>'
            st.markdown(href, unsafe_allow_html=True)

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        pdf.multi_cell(0, 10, f"ROC-AUC Score: {roc_auc_score(y_test, proba):.2f}")
        pdf.multi_cell(0, 10, f"Classification Report:\n{classification_report(y_test, y_pred)}")

        pdf_path = "breast_cancer_diagnosis_report.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="diagnosis_report.pdf">📥 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)