import streamlit as st
import pandas as pd
import pickle
import os

# 🌐 Streamlit Page Config
st.set_page_config(page_title="EduInsight - Student Predictor", layout="centered")

# 🎓 App Title
st.title("🎓 EduInsight - Student Performance Predictor")
st.markdown("""
Upload a CSV file with the following columns:
- `Attendance`
- `Quiz1_Score`
- `Quiz2_Score`
- `Assignment_Score`
- `LMS_Interactions`
""")

# ✅ Model File Check
model_file = "student_model.pkl"
if not os.path.exists(model_file):
    st.error("⚠️ Model file not found! Please place `student_model.pkl` in the same folder as `app.py`.")
    st.stop()

# 🔍 Load the model
with open(model_file, "rb") as f:
    model = pickle.load(f)

# 📤 CSV File Upload
uploaded_file = st.file_uploader("📁 Upload your student data CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # 🧾 Read and show data
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(input_df)

        # ✅ Make predictions
        preds = model.predict(input_df)

        # 🎯 Label Mapping
        label_map = {0: "At Risk", 1: "Average", 2: "High Performer"}
        input_df["Predicted Performance"] = [label_map.get(p, "Unknown") for p in preds]

        # 📊 Show final result
        st.subheader("🎯 Prediction Results")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a CSV file to see predictions.")
