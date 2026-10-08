# 🚗 EV Battery Degradation & Range Prediction

A machine learning project for analyzing electric vehicle data and predicting **EV driving range** based on vehicle, battery, and driving-related features.

The project includes data preprocessing, exploratory data analysis, machine learning model training, model saving, and a **Streamlit web application** for interactive predictions.

## 🚀 Live Demo

Try the deployed application:

https://ev-battery-degradation-range-prediction-cchtxkovxp8gjvsjdf5amk.streamlit.app/

---

## 📌 Project Overview

Electric vehicle driving range can vary depending on several factors, including battery capacity, battery health, vehicle mileage, temperature, average speed, and charging cycles.

This project uses machine learning to analyze these factors and build a model that can estimate the **driving range of an electric vehicle in kilometers**.

The trained model is integrated with a Streamlit application, allowing users to enter vehicle and battery information and receive a predicted driving range.

---

## 🎯 Objectives

* Analyze electric vehicle and battery-related data.
* Perform data preprocessing and exploratory data analysis.
* Identify relevant features for EV range prediction.
* Train a machine learning regression model.
* Save the trained model and preprocessing objects for reuse.
* Build an interactive Streamlit application.
* Deploy the application for online predictions.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**

---

## 🤖 Machine Learning

The project uses a **Random Forest Regression** model to predict EV driving range.

### Features Used

The model uses vehicle and battery-related features such as:

* Battery Capacity
* Battery Health
* Mileage
* Temperature
* Average Speed
* Charge Cycles
* Vehicle Make
* Drive Type

### Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Preprocessing
     ↓
Feature Selection
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Random Forest Regression
     ↓
Model Evaluation
     ↓
Save Model & Scaler
     ↓
Streamlit Prediction App
```

---

## 📁 Project Structure

```text
EV-Battery-Degradation-Range-Prediction/
│
├── app/
│   └── main.py
│
├── data/
│   └── electricvehicleanalytics.csv
│
├── model/
│   ├── le_drive.pkl
│   ├── le_make.pkl
│   ├── model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── EV_Battery_Analysis.ipynb
│   ├── predict.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The project uses the following dataset:

```text
data/electricvehicleanalytics.csv
```

The dataset contains electric vehicle information related to vehicle characteristics, battery performance, driving conditions, and other factors that can influence EV driving range.

---

## 🔍 Exploratory Data Analysis

Exploratory data analysis was performed using **Pandas, Matplotlib, and Seaborn** to understand the dataset and identify relationships between vehicle/battery features and driving range.

The analysis includes:

* Data inspection
* Missing-value analysis
* Feature analysis
* Distribution analysis
* Relationship analysis
* Data visualization

The complete analysis and experimentation can be found in:

```text
src/EV_Battery_Analysis.ipynb
```

---

## 🧠 Model Training

The model training pipeline is available in:

```text
src/train_model.py
```

The training process includes:

1. Loading the dataset.
2. Cleaning and preprocessing the data.
3. Selecting relevant features.
4. Encoding categorical features.
5. Splitting the data into training and testing sets.
6. Scaling numerical features using `StandardScaler`.
7. Training a Random Forest Regression model.
8. Saving the trained model and preprocessing objects.

---

## 🔮 Prediction

Prediction functionality is available in:

```text
src/predict.py
```

The saved model and preprocessing objects are loaded to generate predictions for new EV data.

---

## 💾 Saved Model Artifacts

The trained machine learning artifacts are stored in the `model/` directory.

| File           | Purpose                                |
| -------------- | -------------------------------------- |
| `model.pkl`    | Trained Random Forest regression model |
| `scaler.pkl`   | Feature scaling object                 |
| `le_make.pkl`  | Vehicle make encoder                   |
| `le_drive.pkl` | Drive-type encoder                     |

These files allow the application to make predictions without retraining the model every time.

---

## 🌐 Streamlit Application

The interactive web application is available in:

```text
app/main.py
```

Users can enter EV-related information through the interface and receive an estimated driving range.

### Run the application locally

```bash
streamlit run app/main.py
```

The application will open in your browser.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yagnasreekamireddy/EV-Battery-Degradation-Range-Prediction.git
```

### 2. Navigate into the project

```bash
cd EV-Battery-Degradation-Range-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app/main.py
```

---

## 🚀 Future Improvements

The project can be further improved by:

* Adding larger and more diverse EV datasets.
* Improving model accuracy through hyperparameter tuning.
* Comparing Random Forest with other regression algorithms.
* Adding dedicated **Battery State of Health (SoH)** prediction.
* Adding **Remaining Useful Life (RUL)** prediction.
* Adding more interactive visualizations.
* Displaying model performance metrics in the application.
* Adding prediction confidence or uncertainty estimates.
* Improving the Streamlit user interface.
* Adding real-time EV battery monitoring data.

---

## 👨‍💻 Author

**Yagna Sree Kamireddy**

GitHub:

https://github.com/yagnasreekamireddy
