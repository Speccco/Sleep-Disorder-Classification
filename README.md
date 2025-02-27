# Streamlit Classification Model App

## 📌 Overview
This is a **Streamlit-powered web application** that allows users to **input features manually** and get real-time predictions from a **pre-trained classification model**. The app also includes **interactive visualizations** to explore relationships within the dataset.

## 🎯 Features
- **User-friendly interface** to input feature values and get predictions.
- **Loads a pre-trained classification model (`model.pkl`)**.
- **Data Visualizations Tab** for insights:
  - Age vs. Sleep Disorder (categorized by Gender)
  - Quality of Sleep vs. Stress Level (Regression Plot)
  - BMI Category vs. Stress Level
  - Sleep Duration vs. Stress Level
  - Crosstab analysis of Gender, Occupation, Sleep Disorder, and BMI Category.

## 🚀 How to Run the App
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/streamlit-classification-app.git
cd streamlit-classification-app
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```

## 📂 Project Structure
```
streamlit-classification-app/
│── app.py                   # Main Streamlit App Script
│── model.pkl                # Pre-trained Classification Model
│── requirements.txt         # Required Python packages
│── Sleep_health_and_lifestyle_dataset.csv # Dataset for visualizations
│── README.md                # Project Documentation (This file)
```

## 📊 Visualizations in the App
The app provides **interactive plots** generated using **Seaborn and Matplotlib**:
- **Bar Plots**: Age vs. Sleep Disorder (by Gender), BMI vs. Stress Level
- **Regression Plots**: Sleep Duration vs. Stress Level
- **Crosstab Analysis**: Gender, Occupation, and Sleep Disorders

## 🤝 Contributing
Feel free to contribute to this project by improving UI, adding more models, or enhancing visualizations. Fork, modify, and submit a **pull request**.

## 🛠️ Technologies Used
- **Python** (pandas, numpy, scikit-learn)
- **Streamlit** (for web UI)
- **Seaborn & Matplotlib** (for visualizations)
- **pickle** (for model persistence)

## 📜 License
This project is licensed under the MIT License - feel free to modify and use it!


