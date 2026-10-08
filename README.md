\# 🚗 EV Battery Degradation \& Range Prediction



A machine learning project for predicting electric vehicle battery degradation and estimating EV driving range based on vehicle and battery-related features.



\## 📌 Project Overview



Electric vehicle battery performance decreases over time due to factors such as battery usage, vehicle characteristics, driving conditions, and other operational parameters.



This project uses machine learning to analyze electric vehicle data and build a predictive model for EV-related performance.



\## 🎯 Objectives



\- Predict EV battery degradation/performance.

\- Estimate electric vehicle driving range.

\- Process and analyze EV datasets.

\- Train a machine learning model.

\- Use trained models to make predictions.

\- Provide predictions through an application interface.



\## 🛠️ Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- Streamlit

\- Matplotlib

\- Seaborn

\- Jupyter Notebook



\## 📁 Project Structure



EV-Battery-Degradation-Range-Prediction/ │ ├── app/ │ └── main.py │ ├── data/ │ └── electricvehicleanalytics.csv │ ├── model/ │ ├── ledrive.pkl │ ├── lemake.pkl │ ├── model.pkl │ └── scaler.pkl │ ├── src/ │ ├── Untitled.ipynb │ ├── predict.py │ └── train\_model.py │ ├── .gitignore ├── README.md └── requirements.txt





\## ⚙️ Installation



Clone the repository:



git clone https://github.com/yagnasreekamireddy/EV-Battery-Degradation-Range-Prediction.git





Navigate into the project:



cd EV-Battery-Degradation-Range-Prediction





Create a virtual environment:



python -m venv venv





Activate it on Windows:



venv\\Scripts\\activate





Install the required packages:



pip install -r requirements.txt





\## ▶️ Running the Application



Run the Streamlit application:



streamlit run app/main.py





The application will provide a browser-based interface for making EV predictions.



\## 🤖 Machine Learning



The project contains trained machine learning artifacts in the `model/` directory.



These include:



\- `model.pkl` — trained prediction model

\- `scaler.pkl` — feature scaling object

\- `le\_make.pkl` — vehicle make encoder

\- `le\_drive.pkl` — drive-type encoder



The training pipeline is available in:



src/train\_model.py





Prediction functionality is available in:



src/predict.py





\## 📊 Dataset



The project uses:



data/electricvehicleanalytics.csv





The dataset contains electric vehicle-related information used for analysis and machine learning.



\## 📓 Jupyter Notebook



The exploratory analysis and experimentation can be found in:



src/Untitled.ipynb





\## 🚀 Future Improvements



\- Improve model accuracy using additional EV and battery datasets.

\- Add battery State of Health (SoH) prediction.

\- Add battery Remaining Useful Life (RUL) prediction.

\- Compare multiple machine learning algorithms.

\- Add interactive data visualizations.

\- Deploy the application online.

\- Add model performance metrics and prediction confidence.



\## 👨‍💻 Author



\*\*Yagna Sree Kamireddy\*\*



GitHub:  

https://github.com/yagnasreekamireddy

