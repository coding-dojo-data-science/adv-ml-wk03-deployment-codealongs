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
import plotly.io as pio

## Functions

## load_data
# @st.cache_data
def load_data():
    df = pd.read_csv('../Data/loan_approval.csv')
    return df

## explore_categorical (Copied from LP)
def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
 
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # Before filling nulls, save null value counts and percent for printing 
  null_count = temp_df[x].isna().sum()
  null_perc = null_count/len(temp_df)* 100
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # Create figure with desired figsize
  fig, ax = plt.subplots(figsize=figsize)
  # Plotting a count plot 
  sns.countplot(data=temp_df, x=x, ax=ax, order=order)
  # Rotate Tick Labels for long names
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title with the feature name included
  ax.set_title(f"Column: {x}")
  
  # Fix layout and show plot (before print statements)
  fig.tight_layout()
  plt.show()
    
  return fig

## explore_numeric (Copied from LP)
def explore_numeric(df, x, figsize=(6,5) ):
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}")
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()
  
  # Ensure plot is shown before message
  plt.show()
  ## Print message with info on the count and % of null values
  null_count = df[x].isna().sum()
  null_perc = null_count/len(df)* 100
  print(f"- NaN's Found: {null_count} ({round(null_perc,2)}%)")
  return fig

## Global Variables

## Data
df = load_data()

## Columns for EDA
columns = df.columns
features = [col for col in columns if col != 'loan_status']
target = 'loan_status'

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

## selectbox for columns
eda_column = st.sidebar.selectbox('Column to Explore', columns, index=None)

## Conditional: if column was chosen
if eda_column:
    ## Check if column is object or numeric and use appropriate plot function
    if df[eda_column].dtype == 'object':
        fig = explore_categorical(df, eda_column)
    else:
        fig = explore_numeric(df, eda_column)

    ## Show plot
    st.subheader(f'Display Descriptive Plots for {eda_column}')
    st.pyplot(fig)

## Feature vs Target

feature_vs_target = st.sidebar.selectbox('Compare Feature to Target', features, index=None)

if feature_vs_target:
    ## Check if feature is numeric or object
    if df[feature_vs_target].dtype == 'object':
        comparison = df.groupby('loan_status').count()
        title = f'Count of {feature_vs_target} by {target}'
    else:
        comparison = df.groupby('loan_status').mean()
        title = f'Mean {feature_vs_target} by {target}'

    ## Display appropriate comparison
    pfig = px.bar(comparison, y=feature_vs_target, title=title)
    st.plotly_chart(pfig)
        
    

    
