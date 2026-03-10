# Calories Prediction Project
- A machine learning–based web application that predicts **calorie expenditure** using user health and activity parameters.  
- This project demonstrates a complete **end-to-end ML workflow** — from data preprocessing and model training to deployment using Streamlit.

# Project Overview
- The goal of this project is to estimate the number of calories burned during physical activity based on physiological and workout-related inputs.  
- The trained regression model is integrated into an interactive **Streamlit web application** that provides real-time predictions.

# Input Features
- The model uses the following inputs:
  - Gender
  - Age
  - Height (cm)
  - Weight (kg)
  - Exercise Duration (minutes)
  - Heart Rate (bpm)
  - Body Temperature (°C)

# Output
 **Predicted Calories Burned**

# Machine Learning Workflow
1. **Data Collection & Cleaning**
   - Loaded and validated the dataset
   - Ensured correct data types and handled inconsistencies

2. **Exploratory Data Analysis**
   - Analyzed feature distributions
   - Skiwness

3. **Feature Engineering**
   - Selected relevant numerical and categorical features
   - Scaled numerical values where necessary

4. **Model Training**
   - Trained a regression model for calorie prediction
   - Evaluated model performance using metrics such as:
     - R² Score
     - Mean Absolute Error (MAE)

5. **Model Serialization**
   - Saved the trained model using `pickle`

6. **Application Development**
   - Built a Streamlit UI for user interaction
   - Integrated the trained model for live predictions


## 📂 Project Structure

Calories-Prediction-Project/
│
├── app.py # Streamlit web application

├── dataset.csv # Dataset used for model training

├── model.pkl # Trained machine learning model

├── requirements.txt # Project dependencies

└── README.md # Project documentation

# Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **Streamlit**
- **Git & GitHub**

# How to Run the Project Locally

# Step 1: Clone the repository
 git clone https://github.com/rohanyadav782/Calories-Prediction-Project.git

# Step 2: Navigate to the project directory
- cd Calories-Prediction-Project

# Step 3: Install required dependencies
- pip install -r requirements.txt

# Step 4: Run the Streamlit application
streamlit run app.py OR python -m streamlit run app.py
