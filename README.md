# Breast_Cancer_Detection

📊 Dataset Summary: breast_cancer_data.csv <br>
This dataset contains 569 rows and 33 columns, with no missing values in meaningful features (except for one column).<br>

✅ Data Quality<br>
No missing values in any of the main features.<br>

Column Unnamed: 32 has all NaN values → Safe to drop.<br>

Column id is just an identifier → Not useful for modeling.<br>

🏷️ Target Column<br>
diagnosis (Target):<br>

Type: object (Categorical)<br>

Values:<br>

M = Malignant (cancerous)<br>

B = Benign (non-cancerous)<br>

📐 Feature Columns<br>
The rest of the columns are numerical features derived from medical images (WBC nuclei features) — categorized into:<br>

Category	Example Columns<br>
Mean	radius_mean, texture_mean, ...<br>
Standard Error (SE)	radius_se, texture_se, ...<br>
Worst (Max)	radius_worst, area_worst, ...<br>

Each type captures different statistical properties of the tumor.<br>

📌 Descriptive Statistics (Highlights)<br>
Feature	Min	Max	Mean	Description<br>
radius_mean	6.98	28.11	14.13	Average size of the tumor radius<br>
area_mean	143.5	2501.0	654.89	Tumor surface area<br>
concavity_mean	0.00	0.43	0.088	Severity of concave portions<br>
fractal_dimension	0.05	0.20	0.08	Complexity of tumor border<br>

📌 To Do (Preprocessing Plan)<br>
Task	Action<br>
Drop id and Unnamed: 32	Not useful for ML<br>
Convert diagnosis to binary	M=1, B=0<br>
Normalize/standardize features	Due to different scales<br>
Split into Train/Test	For model evaluation<br>

<b> Install Required Module </b><br>

```bash
pip install -r requirement.txt
```

<b> For Running app.py </b><br>

```bash
pip install stremalit
```

``` bash
streamlit run app.py
```
<b>Project Structure</b>
Breast_Cancer_Detection/<br>
├── app.py<br>
├── breast_cancer_data.csv<br>
├── requirements.txt<br>
└── README.md   (optional but recommended)<br>

📈 What is ROC-AUC? <br>
✅ ROC – Receiver Operating Characteristic Curve<br>
The ROC Curve is a graph showing the trade-off between:<br>

True Positive Rate (TPR) → Also called Sensitivity or Recall<br>

False Positive Rate (FPR) → The proportion of healthy cases wrongly classified as positive<br>

The ROC curve plots TPR vs FPR at various classification thresholds.<br>

🧠 AUC – Area Under the Curve<br>
The AUC (Area Under Curve) quantifies the overall ability of the model to distinguish between the classes (Benign vs Malignant). <br>

AUC is a single scalar value between 0 and 1.<br>

AUC Score	Interpretation<br>
0.90 – 1.0	Excellent model<br>
0.80 – 0.90	Good model<br>
0.70 – 0.80	Fair model<br>
0.60 – 0.70	Poor model<br>
0.50	No better than random guessing<br>

💡 In This Breast Cancer App:<br>
A high ROC-AUC means the model can effectively separate malignant from benign tumors.<br>

Unlike just accuracy, ROC-AUC considers class imbalance and is robust in medical diagnosis contexts.<br>