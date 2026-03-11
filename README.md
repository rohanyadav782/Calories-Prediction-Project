# Calories Prediction Project

A machine learning–powered web application that predicts calorie expenditure using user-specific health and activity data.
This project represents a complete end-to-end ML pipeline, covering data analysis, model training, secure backend integration, and cloud deployment using Streamlit.

# Project Overview

The objective of this project is to accurately estimate the number of calories burned during physical activity based on physiological and workout-related parameters.
A trained regression model is embedded into an interactive web application that delivers real-time predictions while securely storing user inputs and results in a PostgreSQL database.
This project demonstrates how machine learning models can be transformed into production-ready applications.

# Input Features
The prediction model uses the following user inputs:
- Gender
- Age
- Height (cm)
- Weight (kg)
- Exercise Duration (minutes)
- Heart Rate (bpm)
- Body Temperature (°C)

# Output
Predicted Calories Burned

# Machine Learning Workflow
 1.Data Collection & Validation:
   - Loaded and verified the dataset
   - Ensured correct data types and data consistency

 2.Exploratory Data Analysis (EDA):
   - Analyzed feature distributions
   - Identified skewness and patterns affecting calorie expenditure

 3.Feature Engineering:
   - Selected relevant numerical and categorical features
   - Applied feature scaling for improved model performance

 4.Model Training & Evaluation:
   - Trained a regression-based machine learning model 
   - Evaluated performance using:
     - R² Score
     - Mean Absolute Error (MAE)

 5.Model Serialization:
   - Saved the trained model using pickle for reuse in deployment

 6.Application Development:
   - Built an interactive UI using Streamlit
   - Integrated the trained model for live predictions
   - Implemented secure environment variable handling for credentials
   - Stored prediction results in PostgreSQL(Cloud)

# Project Structure
Calories-Prediction-Project/
│
├── app.py              # Streamlit web application
├── dataset.csv         # Dataset used for model training
├── model.pkl           # Serialized ML model
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

# Tech Stack
 - Python
 - Pandas & NumPy
 - Scikit-learn
 - Matplotlib & Seaborn
 - Streamlit
 - PostgreSQL
 - Git & GitHub

# How to Run the Project Locally
 - Step 1: Clone the repository
   git clone https://github.com/rohanyadav782/Calories-Prediction-Project.git

 - Step 2: Navigate to the project directory
   cd Calories-Prediction-Project
 - Step 3: Install required dependencies
   pip install -r requirements.txt
 - Step 4: Run the Streamlit application
   streamlit run app.py OR python -m streamlit run app.py
   
This description now screams:

✅ Production-ready
