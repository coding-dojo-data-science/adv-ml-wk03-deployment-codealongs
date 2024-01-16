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

    
## TODO 

## load_data() with caching

@st.cache_data


## Global Variables

## Data


## TODO

## Columns for EDA


## TODO

## Image, title and Markdown subheader



## TODO

## Display DataFrame


## TODO

## Display .info()
## Get string for .info()






## TODO

## Descriptive Statistics Subheader


## TODO

## Button for Statistics




## TODO

## Eda Plots subheader


## TODO

## selectbox for columns
# ## Eda Plots

eda_column = st.sidebar.selectbox('Column to Explore', columns, index=None)

## Conditional: if column was chosen
if eda_column:
    ## Check if column is object or numeric and use appropriate plot function
    
    

    ## Show plot


## Feature vs Target

feature_vs_target = st.sidebar.selectbox('Compare Feature to Target', features, index=None)

if feature_vs_target:
    ## Check if feature is numeric or object
    if df[feature_vs_target].dtype == 'object':
        none
            
    else:
        none

    ## Display appropriate comparison
    pfig = px.bar(comparison, y=feature_vs_target, title=title)
    
    st.plotly_chart(pfig)
        


