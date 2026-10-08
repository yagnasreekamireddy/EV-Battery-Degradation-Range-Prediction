🚗 EV Battery Degradation & Range Prediction

A machine learning project for predicting electric vehicle battery degradation and estimating EV driving range based on vehicle and battery-related features.

🚀 Live Demo

Try the deployed application:
https://ev-battery-degradation-range-prediction-cchtxkovxp8gjvsjdf5amk.streamlit.app/

📌 Project Overview

Electric vehicle battery performance decreases over time due to factors such as battery usage, vehicle characteristics, driving conditions, and other operational parameters.

This project uses machine learning to analyze electric vehicle data and build a predictive model for EV-related performance.

The trained model is integrated with a Streamlit web application that provides a simple interface for making predictions.

🎯 Objectives
Predict EV battery degradation/performance.
Estimate electric vehicle driving range.
Process and analyze EV datasets.
Train a machine learning model.
Use trained models to make predictions.
Provide predictions through an interactive web application.
🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
Matplotlib
Seaborn
Jupyter Notebook
📁 Project Structure
EV-Battery-Degradation-Range-Prediction/
│
├── app/
│   └── main.py
│
├── data/
│   └── electricvehicleanalytics.csv
│
├── model/
│   ├── ledrive.pkl
│   ├── lemake.pkl
│   ├── model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── Untitled.ipynb
│   ├── predict.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt

⚙️ Installation
1. Clone the repository
git clone https://github.com/yagnasreekamireddy/EV-Battery-Degradation-Range-Prediction.git

2. Navigate into the project
cd EV-Battery-Degradation-Range-Prediction

3. Create a virtual environment
python -m venv venv

4. Activate the virtual environment

On Windows:

venv\Scripts\activate

5. Install the required packages
pip install -r requirements.txt

▶️ Running the Application

Run the Streamlit application:

streamlit run app/main.py


The application will open in your browser and provide an interactive interface for making EV predictions.

🤖 Machine Learning

The project contains trained machine learning artifacts in the model/ directory.

These include:

model.pkl — trained prediction model
scaler.pkl — feature scaling object
le_make.pkl — vehicle make encoder
le_drive.pkl — drive-type encoder

The model training pipeline is available in:

src/train_model.py


Prediction functionality is available in:

src/predict.py

📊 Dataset

The project uses the following dataset:

data/electricvehicleanalytics.csv


The dataset contains electric vehicle-related information used for data analysis and machine learning.

📓 Jupyter Notebook

Exploratory data analysis and experimentation can be found in:

src/EV_Battery_Analysis.ipynb

🚀 Future Improvements
Improve model accuracy using additional EV and battery datasets.
Add battery State of Health (SoH) prediction.
Add battery Remaining Useful Life (RUL) prediction.
Compare multiple machine learning algorithms.
Add interactive data visualizations.
Add model performance metrics and prediction confidence.
Improve the user interface and visualization of prediction results.
👨‍💻 Author

Yagna Sree Kamireddy

GitHub:
https://github.com/yagnasreekamireddy
