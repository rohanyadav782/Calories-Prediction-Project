
# Dataset: Calories - Regression Techniques

# --- Import required libraries ---

# --- For data manipulation ---
import pandas as pd
import numpy as np

# --- To store trained model --- 
import pickle

# --- For visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- For model building ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- For evaluation ---
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load dataset ---
data = pd.read_csv(r"C:\Users\91788\Desktop\Python\Data Science\linear regression\Calorie_Prediction_Project\main_file\dataset.csv")

# --- Display basic information ---
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# --- Removed Unique-ID and Target, dropping catogerical feature ---
df = data.drop(columns= ["id","Sex","Calories"])
target_df = data["Calories"]
    
# --- Function to clean data ---
# 1 - Handling null value  
# 2 - Treating Outliers
def data_cleaning(feature,dataset):
    
    # --- IQR Method to find outliers ---
    q1,q3 = np.percentile(dataset[feature],[25,75])
    IQR = q3-q1
    lower_fence = q1-1.5*(IQR)
    upper_fence = q3+1.5*(IQR)
    skew = dataset[feature].skew()
    
    # --- Treating outlier & null value based on feature skewness --- 
    if skew >= -0.5 and skew <= 0.5:
        mean_value = dataset[feature].mean()
        dataset[feature].fillna(mean_value, inplace=True)
        dataset.loc[(dataset[feature] < lower_fence) | (dataset[feature] > upper_fence),feature] = mean_value
        print(f"outlier removed for {feature}")

    elif skew > 0.5 or skew < -0.5 or skew >=1 or skew <-1:
        median_value = dataset[feature].median()
        dataset[feature].fillna(median_value, inplace=True)
        dataset.loc[(dataset[feature] < lower_fence) | (dataset[feature] > upper_fence),feature] = median_value
        print(f"outlier removed for {feature}")
    
    # --- Plotting outlier free box-plot for each feature ---
    sns.boxplot(dataset[feature])
    plt.show()
    return dataset
   
# --- Iterating over each feature to treat null value & outliers --- 
for x in df.columns:
    sns.boxplot(df[x])
    print(x,"skewed value :-",df[x].skew())
    plt.show()
    data_cleaned = data_cleaning(x,df)
    
# --- Train-Test-Split ---
Sample_train, Sample_test, Target_train, Target_test = train_test_split(
    data_cleaned, target_df , test_size=0.2, random_state=42)

# --- Feature Scaling ---
scaler = StandardScaler()
Sample_train_scaled = scaler.fit_transform(Sample_train)
Sample_test_scaled = scaler.transform(Sample_test)

# --- Train Model ---
model = LinearRegression()
model.fit(Sample_train_scaled, Target_train)

# --- Predictions ---
# Training
prediction_train = model.predict(Sample_train_scaled)
r2_train = r2_score(Target_train, prediction_train)*100

# --- Testing (MODEL EVALUATION)---
prediction_test = model.predict(Sample_test_scaled)  
r2_test = round(r2_score(Target_test, prediction_test)*100,2)
mae = round(mean_absolute_error(Target_test, prediction_test),2)
mse = round(mean_squared_error(Target_test, prediction_test),2)
rmse = round(np.sqrt(mse),2)

print("Model Evaluation Results")
print("-------------------------")
print("R2 Score :", r2_test)
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)

# --- Relative Diff (Overfitting) ---
relative_diff = round((r2_test-r2_train)/r2_test,)
print("Overfitting score :",relative_diff)

# --- Model Coefficients (Feature Importance) ---
coeff_df = pd.DataFrame({
    "Feature": data_cleaned.columns,
    "Co-efficient": model.coef_}).sort_values(by="Co-efficient", ascending=False)
print(coeff_df)
coeff_df.to_excel("Feature_co-efficient.xlsx",index = False)

# --- Bar graph for feature co-efficient ---
coeff_df.plot(
    x="Feature",
    y="Co-efficient",
    kind="bar")
plt.title("Feature Co-efficient (Linear Regression)")
plt.show()

# --- Intercept ---
print("Intercept:", model.intercept_)

# --- Normalizing Mean AVG error(Calculate error ) ---
error = mae/np.mean(target_df)  
print(error)

# --- Plot for actual VS prediction 
plt.scatter(Target_test, prediction_test)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted")
plt.show()

# --- Predicting on new data --- 
new_data = np.array([[30, 175, 70, 45, 110, 38]])
new_data_scaled = scaler.transform(new_data)
predictionnew = model.predict(new_data_scaled)
print("Predicted Calories Burned:", predictionnew[0])

def about_project():
    return """

            PREDICT CALORIE EXPENDITURE - ML PROJECT


\t--- Project Overview ---
This project predicts the amount of calories burned during
physical activity using Machine Learning techniques.
It analyzes health and exercise-related parameters to
estimate calorie expenditure accurately.

\t --- Problem Statement ---
To develop a machine learning model that can predict
calorie expenditure based on physiological and
activity-related features such as age, weight, height,
exercise duration, heart rate, and body temperature.

\t --- Dataset Information ---
Dataset Source : Syntetic
Dataset Name   : Predict Calorie Expenditure
Total Features : 8
Target Column  : Calories

\t --- Features Used in the Model ---
- Age
- Height
- Weight
- Duration (Exercise Time)
- Heart Rate
- Body Temperature
- Gender

\t --- Technologies & Tools Used ---
- Python
- Pandas & NumPy (Data Processing)
- Matplotlib & Seaborn (Data Visualization)
- Scikit-learn (Machine Learning Model)
- Pickle (Model Serialization)
- PostgreSQL (Database Storage)

\t --- Machine Learning Workflow ---
1. Data Collection 
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Scaling using StandardScaler
5. Model Training using Linear Regression
6. Model Evaluation using different metrics
7. Model Saving using Pickle file
8. Menu Driven Interface for User Interaction

\t --- Model Used ---
- Linear Regression

\t --- Model Evaluation Metrics ---
- R2 Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

\t --- Key Features of the Project ---
- Menu Driven Console Interface
- Automatic Dataset Statistical Summary
- Data Visualization Support
- Real-time Calorie Prediction
- Structured Machine Learning Pipeline

\t --- Future Scope ---
- Integrate with Smart Fitness Devices
- Develop a Mobile Fitness Tracking Application
- Add real-time health monitoring features

\t --- Purpose of the Project ---
\nThis project demonstrates the complete lifecycle of a
Machine Learning project including data preprocessing,
exploratory analysis, model training, evaluation,
and deployment using Python.

Developed By :
\nRohan Yadav
Data Science Project

"""
# --- storing require contents in pickle file ---
model_data = {
    "model": model,
    "scaler": scaler,
    "data":data_cleaned,
    "model_evaluation":pd.DataFrame({"R2 Score" : r2_test,
    "Mean Absolute Error" : mae,
    "Mean Squared Error" : mse,
    "Root Mean Squared Error": rmse,
    "Overfitting Score" : relative_diff},index=[0]),
    "Feature_Co-efficient" : coeff_df,
    "About_Project" : about_project()}

# --- save into pickle file ---
with open("model.pkl", "wb") as file :
    pickle.dump(model_data, file)

print("Model and Scaler saved successfully!")
