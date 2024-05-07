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


## Define load_data function with caching


## Use function to load Data



## Image, title and Markdown subheader


## Display DataFrame



## .info()
## Get string for .info()


#Display .info() with button trigger


## Descriptive Statistics subheader


## Button for Statistics


## Eda Plots subheader


## Columns for EDA



## selectbox for columns


## Conditional: if column was chosen

    ## Check if column is object or numeric and use appropriate plot function

    ## Show plot


## Select box for features vs target


## Conditional: if feature was chosen

    ## Check if feature is numeric or object


    ## Display appropriate comparison

        
    

    
