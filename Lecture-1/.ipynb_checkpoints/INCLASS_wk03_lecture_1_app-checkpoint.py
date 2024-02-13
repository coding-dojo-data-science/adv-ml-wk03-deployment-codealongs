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
import plotly as px
import functions as fn


## load_data() with caching
@st.cache_data

def load_data():
    df = pd.read_csv('../Data/loan_approval.csv')
    return df
## TODO

## Global Variables

## Data
df = load_data()
## TODO

## Columns for EDA

## TODO

## Image, title and Markdown subheader
st.image('../Images/money_tree.jpg')
st.title('Loan Approval app')

## TODO

## Display DataFrame

## TODO

## .info()
## Get string for .info()

## TODO

## Display .info()

## TODO

## Descriptive Statistics Subheader

## TODO

## Button for Statistics

## TODO

## Eda Plots subheader

## TODO

## selectbox for columns

## TODO

## Conditional: if column was chosen

## TODO

    ## Check if column is object or numeric and use appropriate plot function

    ## TODO

    ## Show plot
    
    ## TODO
    
## Feature vs Target

## Create sidebar button

    ## Check if feature is numeric or object


    ## Display appropriate comparison
