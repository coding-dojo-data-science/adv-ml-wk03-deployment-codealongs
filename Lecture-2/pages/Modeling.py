## Advanced Machine Learning Week 3 Lecture 2
## Objectives:
## Create a multipage app
## That serves model predictions

## imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import streamlit as st
from io import StringIO
import joblib
import functions as fn

## paths
pathlib = joblib.load('pathlib.joblib')

## Load model and data

@st.cache_data
def load_data(pathlib):
    train_path = pathlib['data']['train']
    X_train, y_train =  joblib.load(train_path)
    test_path = pathlib['data']['test']
    X_test, y_test = joblib.load(test_path)
    return X_train, y_train, X_test, y_test

@st.cache_resource
def load_model(pathlib, model_name='RF'):
    model_path = pathlib['models'][model_name]
    model = joblib.load(model_path)
    return model
    
## Load Data
X_train, y_train, X_test, y_test = load_data(pathlib)
labels = ['Approved', 'Rejcted']

## Load Model
model_name = st.sidebar.selectbox('Select Model', ['RF'])
model = load_model(pathlib, model_name=model_name)

st.title('Model Evaluation and Predictions')

## Evaluate Model
if st.sidebar.button('Evaluate Model'):
    st.subheader(f'Evaluation of {model_name}')
    train_report, test_report, eval_fig = fn.eval_classification(model, X_train, y_train, X_test, y_test,
                                                             labels=labels)
    st.text('Training Report')
    st.text(train_report)
    st.text('Testing Report')
    st.text(test_report)
    st.pyplot(eval_fig)

    test_report = classification_report(y_test, model.predict(X_test))

## Inputs
dependents = st.sidebar.slider('Number of Dependents', min_value=0, max_value=5)
graduated = st.sidebar.radio('Graduated College', ['Not Graduate', 'Graduate'])
self_employed = st.sidebar.radio('Self Employed', ['Yes', 'No'])
income_annum = st.sidebar.number_input('Annual Income', min_value=200000, max_value=10000000)
loan_term = st.sidebar.slider('Years to Repay', min_value=2, max_value=20)
cibil_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=900)
res_asset_val = st.sidebar.number_input('Home Value', min_value=0, max_value=30000000)
comm_asset_val = st.sidebar.number_input('Value of Commercial Assets', min_value=0, max_value=20000000)
lux_asset_val = st.sidebar.number_input('Value of Luxury Assets', min_value=0, max_value=40000000)
bank_asset_val = st.sidebar.number_input('Cash in Bank', min_value=0, max_value=20000000)
loan_amount = st.sidebar.number_input('Requested Loan Amount', min_value=300000, max_value=40000000)

## Prediction
if st.sidebar.button('Make Prediction'):
    try:
        user_data = pd.DataFrame([[dependents, graduated, self_employed, income_annum, loan_amount, loan_term, cibil_score, res_asset_val, comm_asset_val,
                                   lux_asset_val, bank_asset_val]], columns=X_train.columns)
        
        prediction = model.predict(user_data)[0]
        if prediction == 'Rejected':
            color = "red"
        else:
            color = "green"
            
        st.markdown(f'# <span style="color:{color}"> Loan {prediction} </span>',
                   unsafe_allow_html=True)
        
    except Exception as e:
        st.text(e)
        st.header('Please input all required information on the left')


