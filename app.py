import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# PAGE CONFIG
# -----------------------------------------------
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👔", layout="wide")
st.title("👔 Employee Attrition Predictor")

# -----------------------------------------------
# SAMPLE DATA (expanded for stability)
# -----------------------------------------------
data = {
    "Age":[24,32,45,38,55,29,41,36]*5,
    "YearsAtCompany":[1,4,10,6,18,2,8,5]*5,
    "MonthlySalary":[30000,55000,90000,70000,120000,35000,80000,65000]*5,
    "JobSatisfaction":[2,4,3,4,2,3,4,3]*5,
    "WorkLifeBalance":[2,4,3,4,1,3,4,3]*5,
    "OverTime":[1,0,0,0,1,1,0,0]*5,
    "NumProjects":[3,5,6,4,7,3,5,4]*5,
    "TrainingLastYear":[0,2,3,1,1,0,2,1]*5,
    "Attrition":[1,0,0,0,1,1,0,0]*5
}

df = pd.DataFrame(data)

# -----------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------
df["SalaryPerYear"] = df["MonthlySalary"] * 12
df["SatisfactionScore"] = (df["JobSatisfaction"] + df["WorkLifeBalance"]) / 2
df["BurnoutRisk"] = df["OverTime"] * df["NumProjects"] / (df["YearsAtCompany"] + 1)

feature_cols = [c for c in df.columns if c != "Attrition"]

# -----------------------------------------------
# SCALING
# -----------------------------------------------
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

st.dataframe(df.head())

# -----------------------------------------------
# CORRELATION HEATMAP (Matplotlib)
# -----------------------------------------------
st.subheader("Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots()
cax = ax.matshow(corr, cmap="coolwarm")
fig.colorbar(cax)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")

st.pyplot(fig)

# -----------------------------------------------
# MODEL
# -----------------------------------------------
X = df[feature_cols]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

st.subheader("Model Metrics")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.write("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

# -----------------------------------------------
# CONFUSION MATRIX (FIXED ✅)
# -----------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0,1])

fig, ax = plt.subplots()
cax = ax.matshow(cm, cmap="Reds")
fig.colorbar(cax)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(["Stay","Leave"])
ax.set_yticklabels(["Stay","Leave"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i,j], ha="center", va="center")

st.pyplot(fig)

# -----------------------------------------------
# PREDICTION
# -----------------------------------------------
st.subheader("Predict Attrition")

age = st.slider("Age",18,65,35)
yrs = st.slider("Years at Company",0,30,5)
salary = st.number_input("Salary",10000,200000,60000)
job = st.slider("Job Satisfaction",1,4,3)
wlb = st.slider("Work Life Balance",1,4,3)
ot = st.selectbox("Overtime",[0,1])
proj = st.slider("Projects",1,10,5)
train = st.slider("Training",0,5,2)

if st.button("Predict"):

    sal_year = salary * 12
    sat = (job + wlb)/2
    burn = ot * proj / (yrs + 1)

    input_data = [[age, yrs, salary, job, wlb, ot, proj, train,
                   sal_year, sat, burn]]

    input_df = pd.DataFrame(input_data, columns=feature_cols)
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error("⚠️ Employee likely to leave")
    else:
        st.success("✅ Employee likely to stay")