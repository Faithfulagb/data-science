import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px

#set the config page
st.set_page_config(
    page_title= "Bank Churn Prediction",
    page_icon= "💹",
    layout="centered"
)

@st.cache_resource
def load_model_artifact():
    try:
        model_pipeline = joblib.load("bank_churned_model_pipeline.pkl")
        features_columns = joblib.load("bank_churned_feature_columns.pkl")
        numerical_features = [
            "CreditScore",
            "Age",
            "Tenure",
            "EstimatedSalary",
            "Balance",
            "NumOfProducts",
            "ServiceRating"
        ]
        categories_features = [
            "Geography",
            "Gender",
            "HasCrCard",
            "IsActiveMember"
        ]

        return model_pipeline, features_columns
    except FileNotFoundError as e:
       st.error(f"Model not found {e}")
       st.stop()
    except Exception as e:
        st.error(f"An error occured {e}")
        st.stop()       

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("cleaned_bank_customer_churn.csv")
        return df
    except FileNotFoundError as e:
        st.error(f"Dataset not found {e}")
        st.stop()

def predict_churn(cust_data, model_pipeline):
    # try:
    input_df = pd.DataFrame([cust_data])

    expected_columns = [
        "CreditScore", "Geography", "Gender", "Age",
        "Tenure", "EstimatedSalary", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "ServiceRating"
    ]

    input_df = input_df.reindex(columns=expected_columns)

    for col in ["Geography", "Gender"]:
        input_df[col] = input_df[col].astype(str)

    # for col in input_df.columns:
    #     if col not in ["Geography", "Gender"]:
    #         input_df[col] = pd.to_numeric(input_df[col], errors="raise")

    prediction = model_pipeline.predict(input_df)[0]
    proba = model_pipeline.predict_proba(input_df)[0]

    return {
        "prediction": prediction,
        "churn_probability": proba[1],
        "stay_probability": proba[0],
        "result": "Customer will stay" if prediction == 0 else "Customer will churn"
    }

    # except Exception as e:
    #     st.error(f"An error occured {e}")
    #     return None
    
def main():
    st.title("💹 Bank Churn Prediction")
    st.write("Fill all fields to get bank churn prediction")

    #load model artifact
    model_pipeline, features_columns = load_model_artifact()

    #load dataset
    df = load_dataset()

    credit_score = st.number_input("Credit Score")
    geography = st.selectbox("Geography", options=["Germany", "France", "Spain"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=18)
    tenure = st.number_input("Tenure", min_value=1, max_value=50)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
    balance = st.number_input("Balance", min_value=0.0)
    num_of_products = st.number_input("Number Of Products", min_value=1)
    has_cr_card = st.selectbox("Has Credit Card?", options= ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", options= ["Yes", "No"])
    service_rating = st.slider("Service Rating", min_value=1, max_value=5)

    cust_data = {
        "CreditScore" : credit_score,
        "Geography" : geography,
        "Gender" : gender,
        "Age" : age,
        "Tenure" : tenure,
        "EstimatedSalary" : estimated_salary,
        "Balance" : balance,
        "NumOfProducts" : num_of_products,
        "HasCrCard" : has_cr_card,
        "IsActiveMember": is_active_member,
        "ServiceRating" : service_rating

    }

    if st.button("Predict Churn"):
        result = predict_churn(
            cust_data,
            model_pipeline
        )
        if result:
            st.markdown("---") 
            st.subheader(result['result'])
            st.write(f"Churn Probability:{result['churn_probability']: .2%}")
            st.write(f"Stay Probability:{result['stay_probability']: .2%}")

if __name__ == "__main__":
    main()
 