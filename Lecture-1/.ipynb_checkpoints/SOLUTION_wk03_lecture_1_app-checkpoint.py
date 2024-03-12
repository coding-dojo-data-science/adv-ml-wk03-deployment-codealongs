## Advanced Machine Learning Codealong: Introduction to Streamlit
## Week 3: Lecture 1
## Objectives:  
## Create streamlit app to explore a dataset
## Include: Visualize dataframe, print descriptive statistics, and Generate EDA plots of columns

## Reference: https://docs.streamlit.io/library/api-reference

## Import necessary packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import plotly.express as px
import functions as fn

## Functions

## load_data
# @st.cache_data
def load_data():
    df = pd.read_csv('../Data/loan_approval.csv')
    return df

## Global Variables

## Data
df = load_data()



## Image, title and Markdown subheader
st.image('../Images/money_tree.jpg')
st.title('Loan Approval Dataset')
st.markdown("Data gathered from [Kaggle](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)")

## DataFrame
st.header('Loan Approval DataFrame')
st.dataframe(df)

## .info()

## Get info as text
buffer = StringIO()
df.info(buf=buffer)
info_text = buffer.getvalue()

#Display .info() with button trigger
st.sidebar.subheader('Show Dataframe Summary')
summary_text = st.sidebar.button('Summary Text')
if summary_text:
    st.text(info_text)

## Descriptive Statistics
st.sidebar.subheader('Show Descriptive Statistics')

## Button for Statistics
show_stats = st.sidebar.button('Descriptive Statistics')
if show_stats:
    describe = df.describe()
    st.dataframe(describe)

## Eda Plots
st.sidebar.subheader('Explore a Column')


## Columns for EDA
columns = df.columns
target = 'loan_status'
features = [col for col in columns if col != target]


## selectbox for columns
eda_column = st.sidebar.selectbox('Column to Explore', columns, index=None)

## Conditional: if column was chosen
if eda_column:
    ## Check if column is object or numeric and use appropriate plot function
    if df[eda_column].dtype == 'object':
        fig = fn.explore_categorical(df, eda_column)
    else:
        fig = fn.explore_numeric(df, eda_column)

    ## Show plot
    st.subheader(f'Display Descriptive Plots for {eda_column}')
    st.pyplot(fig)

## Feature vs Target

feature = st.sidebar.selectbox('Compare Feature to Target', features, index=None)

if feature:
    ## Check if feature is numeric or object
    if df[feature].dtype == 'object':
        comparison = df.groupby('loan_status').count()
        title = f'Count of {feature} by {target}'
    else:
        comparison = df.groupby('loan_status').mean()
        title = f'Mean {feature} by {target}'

    ## Display appropriate comparison
    pfig = px.bar(comparison, y=feature, title=title)
    st.plotly_chart(pfig)
        
    

    
