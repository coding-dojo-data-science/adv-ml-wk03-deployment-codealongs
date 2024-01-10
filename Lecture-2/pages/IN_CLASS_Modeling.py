## Advanced Machine Learning Week 3 Lecture 2
## Objectives:
## * Create a multipage app
## * That serves model predictions

## imports
import pandas as pd
## imports
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib
import functions as fn
from lime.lime_tabular import LimeTabularExplainer

## paths
pathlib = joblib.load('pathlib.joblib')

## Load model and data

st.title('UNDER CONSTRUCTION') ## REMOVE

@st.cache_data
def load_data(pathlib):
    ## TO DO
    return None
 
@st.cache_resource
def load_model(pathlib, model_name='RF'):
    
    ## TO DO

    return None

@st.cache_resource
def get_explainer(_model_pipe, X_train, labels):
    
    ## TO DO
    return None

@st.cache_resource
def explain_instance(_explainer, _model_pipe, instance_to_explain):
    ## TO DO
    return None

# Header Image

## TO DO

## Page Title

## TO DO

## Load Data

## TO DO

## Load Model

## TO DO

## Evaluate Model Button

## TO DO

## If evaluation button pressed:
## TO DO
    
    ## Evaluate the model
    ## TO DO
    
    ## Report Results
    ## TO DO

## Modeling
## TO DO

## Feature Inputs
dependents = None ## TO DO
graduated = None ## TO DO
self_employed = None ## TO DO
income_annum = None ## TO DO
loan_term = None ## TO DO
cibil_score = None ## TO DO
res_asset_val = None ## TO DO
comm_asset_val = None ## TO DO
lux_asset_val = None ## TO DO
bank_asset_val = None ## TO DO
loan_amount = None ## TO DO

## Prediction Button and Explanation Checkbox
## TO DO
predict = False ## TO DO
explain = False ## TO DO
## If predict button pressed
if predict:
    try:

        ## Add inputs to a dataframe in order of original columns.  Must be 2 dimensional for model.
        pass ## TO DO
        ## Get model prediction
        ## TO DO

        ## Change text color based on model prediction
        ## Red if loan is rejected, Green if it's accepted
        ## TO DO
        
        ## Print model prediction to page using html
        ## TO DO
        
        ## If Explain Prediction checkbox checked
        if explain:
            pass ## TO DO

    ## Catch errors
    except Exception as e:
        st.text(e)
        st.header('Please input all required information on the left')
