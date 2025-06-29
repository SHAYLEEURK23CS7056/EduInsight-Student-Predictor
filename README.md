#  EduInsight – Student Performance Predictor

EduInsight is a web-based machine learning application that predicts student performance based on academic and behavioral input features. Built using Python and Streamlit, it helps educators and institutions identify students who may need academic support.

---

##  Features

- Upload a CSV file with student data
- Predict performance outcomes using a trained ML model
- Simple and clean user interface
- Real-time prediction with no coding required

---

##  Input Features

Make sure your input CSV includes the following columns:

- `Attendance`
- `Quiz1_Score`
- `Quiz2_Score`
- `Assignment_Score`
- `LMS_Interactions`

A sample file (`students.csv`) is included for reference.

---

## 🛠️ Tech Stack / Tools Used

- **Python**
- **Streamlit** – for frontend UI
- **Scikit-learn** – ML model training
- **Pandas** – data handling
- **Pickle** – for model serialization
- **VS Code / Jupyter Notebook** – development environment
- **Git & GitHub** – version control and collaboration

---

##  How to UseClone 
1. Clone the repository or download the files
2. Make sure `student_model.pkl` is in the same folder as `app.py`
3. Run the app:
   ```bash
   streamlit run app.py



