# --- IMPORTING REQUIRED LIBRARIES ---
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

# --- PAGE TITLE ---
st.title("Predict Calorie Expenditure")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        dataset = pickle.load(file)
    return dataset

dataset = load_model()
loaded_model = dataset["model"]
loaded_scaler = dataset["scaler"]
loaded_data = dataset["data"]
about_project = dataset["About_Project"]
feature_coefficient = dataset["Feature_Co-efficient"]
model_evaluation = dataset["model_evaluation"]

# --- DATABASE CONNECTION ---
# def connect_db():
#     conn = psycopg2.connect(
#         host="localhost",
#         database="Predict Calorie Expenditure",
#         user="postgres",
#         password=os.getenv("DB_PASSWORD"),
#         port="5432")
#     return conn

def connect_db():
    conn = psycopg2.connect(
        host="ep-nameless-glade-a1r4rhak-pooler.ap-southeast-1.aws.neon.tech",
        database="predictcalories",
        user="neondb_owner",
        password="npg_aE2r6jWVwTxJ",
        port="5432",
        sslmode="require"
    )
    return conn

conn = connect_db()
cursor = conn.cursor()

# --- SIDEBAR MENU ---
menu = st.sidebar.radio(
    "Navigation",
    [
        "Predict Calories",
        "Data Stats",
        "Database Records",
        "Model Performance",
        "Project Info"
    ])

# --- DATA STATS ---
if menu == "Data Stats":
    st.header("Dataset Statistics")
    stats = []
    for col in loaded_data.columns:
        if col == "index":
            continue

        stats.append({
            "Feature": col,
            "Mean": round(loaded_data[col].mean(),2),
            "Min": round(loaded_data[col].min(),2),
            "Max": round(loaded_data[col].max(),2),
            "Skewness": round(loaded_data[col].skew(),2),
            "Std": round(loaded_data[col].std(),2)
        })
    stats_df = pd.DataFrame(stats)
    st.dataframe(stats_df)

    # --- SKEWNESS GRAPHS ---
    st.subheader("Feature Distribution & Skewness")
    for col in loaded_data.columns:
        if col == "index":
            continue
        fig, ax = plt.subplots()
        sns.histplot(
            loaded_data[col],kde=True,ax=ax )
        ax.set_title(
            f"{col} Distribution (Skewness = {round(loaded_data[col].skew(),2)})")
        st.pyplot(fig)

# --- DATABASE VIEW ---
elif menu == "Database Records":
    st.header("Prediction Database")
    option = st.radio(
        "Select Data Type",
        ["Model Data", "Predicted Data"]    )

    # --- MODEL DATA ---
    if option == "Model Data":
        st.dataframe(loaded_data)

    # --- PREDICTED DATA ---
    elif option == "Predicted Data":
        sub_option = st.radio(
            "Choose Option",
            ["Show Full Records", "Search Specific Record"])

        # --- SHOW FULL RECORDS ---
        if sub_option == "Show Full Records":
            cursor.execute("SELECT * FROM predicted_dataset")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df)

            # --- DOWNLOAD BUTTON ---
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Data",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv")

            # --- PREDICTION HISTORY CHART ---
            st.subheader(" Calories Distribution Analysis ")

            if not df.empty:
                #  --- CREATE CALORIE BINS ---
                bins = [0, 50, 100, 150, 200, 1000]
                labels = ["0-50", "51-100", "101-150", "151-200", "200+"]
                df["calorie_group"] = pd.cut(df["calories"], bins=bins, labels=labels)
                group_counts = df["calorie_group"].value_counts().sort_index()

                # --- BAR CHART ---
                st.subheader("Calories Range Distribution")
                bar_df = group_counts.reset_index()
                bar_df.columns = ["Calories Range", "Count"]
                st.bar_chart(bar_df.set_index("Calories Range"))

                # --- PIE CHART ---
                st.subheader("Calories Percentage Distribution")
                fig, ax = plt.subplots()
                ax.pie(
                    group_counts,
                    labels=group_counts.index,
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.set_title("Calories Burn Distribution")
                st.pyplot(fig)
            else:
                st.warning("No prediction data available")

        # --- SEARCH RECORD ---
        elif sub_option == "Search Specific Record":
            user_id = st.text_input("Enter Person ID").lower().capitalize()
            if st.button("Search"):
                if user_id.strip() == "":
                    st.warning("⚠ Please enter an ID")
                else:
                    query = f"SELECT * FROM predicted_dataset WHERE person_id = '{user_id}'"
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        df = pd.DataFrame(rows, columns=columns)
                        st.success("Prediction Found!")
                        st.dataframe(df)
                    else:
                        st.error("No record found for this ID")

# --- PREDICTION ---
elif menu == "Predict Calories":
    st.header("Predict Calories Burned")
    col1, col2 = st.columns(2)
    with col1:
        person_id = st.text_input("Enter Your ID").lower().capitalize()
        if person_id.strip() == "":
            st.error("Please enter valid ID")
            st.stop()

        else:
            sex = st.selectbox("Gender",["Male","Female"])
            age = st.number_input("Age",20,100)
            height = st.number_input("Height (cm)",110.0,214.0)
            with col2:
                weight = st.number_input("Weight (kg)",35.0,123.0)
                duration = st.number_input("Exercise Duration (minutes)",1,40)
                heart_rate = st.number_input("Heart Rate (BPM)",60.0,125.0)
                body_temp = st.number_input("Body Temperature (C)",26.0,41.5)

                if st.button("Predict Calories"):
                    cursor.execute(
                        "SELECT * FROM predicted_dataset WHERE person_id = %s",
                        (person_id,)
                    )
                    record = cursor.fetchone()
                    if record:
                        st.warning("This user ID already exists in database")
                    else:
                        new_data = np.array([[age,height,weight,duration,heart_rate,body_temp]])
                        new_df = pd.DataFrame(new_data, columns=[
                            "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"
                        ])
                        new_data_scaled = loaded_scaler.transform(new_df)
                        prediction = loaded_model.predict(new_data_scaled)
                        st.success(
                            f"Dear {person_id} you burn something around {round(prediction[0],2)} Calories")

                        cursor.execute("""
                            INSERT INTO predicted_dataset
                            (person_id, sex, age, height, weight, duration, heart_rate, body_temprature, calories)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            person_id,sex,age,height,weight,duration,heart_rate,body_temp,float(prediction[0])))
                        conn.commit()

# --- MODEL PERFORMANCE ---
elif menu == "Model Performance":
    st.header("Model Performance Dashboard")
    st.markdown("Model Accuracy & Error Metrics")
    r2 = model_evaluation["R2 Score"][0]
    mae = model_evaluation["Mean Absolute Error"][0]
    mse = model_evaluation["Mean Squared Error"][0]
    rmse = model_evaluation["Root Mean Squared Error"][0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("R² Score", f"{r2}")
    col2.metric("MAE", f"{mae}")
    col3.metric("MSE", f"{mse}")
    col4.metric("RMSE", f"{rmse}")

    st.subheader("Feature Coefficients")
    fig, ax = plt.subplots()
    feature_coefficient.plot(
        x="Feature",
        y="Co-efficient",
        kind="bar",
        ax=ax)
    ax.set_title("Feature Coefficient (Linear Regression)")
    st.pyplot(fig)

# ---------------- ABOUT PROJECT ----------------
elif menu == "Project Info":
    st.header("Project Description")
    st.write(about_project)