# 💼 Employee Salary Predictor (Streamlit + Machine Learning)

This project predicts the **salary** of an employee based on their academic rank, discipline, gender, and years of experience — using a machine learning model built with **Random Forest Regressor** and a real-world dataset.

It’s a single `.py` file built for simplicity, submission, and demonstration — with **Streamlit** used for a clean web-based interface.

---

## 🧠 How It Works

- Loads a real salary dataset from GitHub automatically
- Encodes categorical features (rank, discipline, gender)
- Trains a Random Forest regression model
- Takes user input and predicts estimated salary 💰

---

## 🚀 How to Run This App

```bash
pip install streamlit pandas scikit-learn requests
streamlit run salary_predictor.py
