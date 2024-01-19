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

#st.title('UNDER CONSTRUCTION') ## REMOVE

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

@st.cache_resource
def get_explainer(_model_pipe, X_train, labels):
    X_train_sc = _model_pipe[0].transform(X_train)
    feature_names = _model_pipe[0].get_feature_names_out()
    explainer = LimeTabularExplainer(
                    X_train_sc,
                    mode='classification',
                    feature_names=feature_names,
                    class_names=labels,
                    random_state=42
                    )
    return explainer



@st.cache_resource
def explain_instance(_explainer, _model_pipe, instance_to_explain):
    instance_to_explain_sc = _model_pipe[0].transform(instance_to_explain)
    explanation = _explainer.explain_instance(instance_to_explain_sc[0],
                                             _model_pipe[-1].predict_proba
                                             )
    return explanation


# Header Image
st.image('Images/money_tree.jpg')
## TO DO

## Page Title
st.title('Model Evaluation and Predictions')
st.subheader('Evaluate the model or make a prediction to the left.')


## TO DO

## Load Data
X_train, y_train, X_test, y_test = load_data(pathlib)
labels = ['Approved', 'Rejected']




## TO DO

## Load Model
model_name = st.sidebar.selectbox('Select Model', ['RF', 'logreg'], index=0)
model = load_model(pathlib, model_name=model_name)



## TO DO

## Evaluate Model Button

st.sidebar.subheader('Evaluation')


## TO DO

## If evaluation button pressed:
## TO DO

if st.sidebar.button('Evaluate model'):
    
    ## Evaluate the model
    ## TO DO
    train_report,test_report, eval_fig = fn.eval_classification(model, X_train,y_train,X_test,y_test, labels = labels)
    ## Report Results
    ## TO DO
    st.text('Training Report')
    st.text(train_report)
    st.text('Testing Report')
    st.text(test_report)
    st.pyplot(eval_fig)


## Modeling
## TO DO
st.sidebar.subheader('Make a Prediction')


## Feature Inputs
dependents = st.sidebar.slider('Number of Dependents', min_value=0, max_value=5)
graduated = st.sidebar.radio('Graduated College', ['Not Graduate', 'Graduate'])
self_employed = st.sidebar.radio('Self Employed', ['Yes', 'No'])
income_annum = st.sidebar.number_input('Annual Income', min_value=20000, max_value=10000000)
loan_term = st.sidebar.slider('Years to Repay', min_value=2, max_value=20)
cibil_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=900)
res_asset_val = st.sidebar.number_input('Home Value', min_value=0, max_value=30000000)
comm_asset_val = st.sidebar.number_input('Value of Commercial Assets', min_value=0, max_value=20000000)
lux_asset_val = st.sidebar.number_input('Value of Luxury Assets', min_value=0, max_value=40000000)
bank_asset_val = st.sidebar.number_input('Cash in Bank', min_value=0, max_value=20000000)
loan_amount = st.sidebar.number_input('Requested Loan Amount', min_value=300000, max_value=40000000)


## Prediction Button and Explanation Checkbox
## TO DO
predict = st.sidebar.button('Make Prediction')
explain = st.sidebar.checkbox('Explain Prediction')


## If predict button pressed
if predict:
    try:
        ## Add inputs to a dataframe in order of original columns.  Must be 2 dimensional for model.
        user_data = pd.DataFrame([[dependents, graduated, self_employed, income_annum, loan_amount, loan_term, cibil_score, res_asset_val, comm_asset_val,
                                   lux_asset_val, bank_asset_val]], columns=X_train.columns)
        ## Get model prediction
        prediction = model.predict(user_data)[0]

        ## Change text color based on model prediction
        ## Red if loan is rejected, Green if it's accepted
        if prediction == 'Rejected':
            color = "red"
        else:
            color = "green"
        
        ## Print model prediction to page using html
        st.markdown(f'# <span style="color:{color}"> Loan {prediction} </span>',
                   unsafe_allow_html=True) # required for html in markdown
        
        ## If Explain Prediction checkbox checked
        if explain:
            explainer = get_explainer(model, X_train=X_train, labels=labels)
            explanation = explain_instance(explainer, model, user_data)
            components.html(explanation.as_html(show_predicted_value=False), 
                            height = 1000
                            )

    ## Catch errors
    except Exception as e:
        st.text(e)
        st.header('Please input all required information on the left')

