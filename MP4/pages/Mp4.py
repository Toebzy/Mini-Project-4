import streamlit as st
import pandas as pd
import numpy as np
import random
import webbrowser
import io
from io import StringIO, BytesIO

import base64
import graphviz
from sklearn import datasets, preprocessing, metrics
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from urllib.error import URLError
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

import streamlit.components.v1 as components
from streamlit.components.v1 import html

import sweetviz as sv


import sys, os
import platform
sys.path.append('../')

st.set_page_config(page_title="MP4 Classification and Clustering", page_icon="📊")

st.title("MP4 Classification and Clustering")
st.sidebar.header("Tabular Data", divider='rainbow')
st.write(
            """This is Group 11's Streamlit MP4 Demo"""
)


# Read the selected file
def readTabData(tab):
        df = pd.read_csv(tab)      
        st.dataframe(df, use_container_width=True)
        return df
    

# Use the analysis function from sweetviz module to create a 'Dataframe Report' object
def eda(df):
    my_report = sv.analyze([df,'EDA'])  
    return my_report
    
def viz1(df):
        st.header('Classification')
        y = st.selectbox('Select y', df.columns) 
        return y


# Prepare data for second vizualisation    
def viz2(df):
        st.header('Dimensions and Measures')
        x = st.selectbox('**Select the first dimension, X**', df.columns)
        z = st.selectbox('**Select the second dimension, Z**', df.columns)
        y = st.selectbox('**Select the measure, Y**', df.columns)     
        return x, y, z

def viz3():
    st.header('predict attrition')
    st.write("Please only use numerical values")
    age = st.number_input('Age', value=0)
    OverTime = st.number_input('OverTime', value=0)
    JobLevel = st.number_input('JobLevel', value=0)
    MaritalStatus = st.number_input('MaritalStatus', value=0)
    MonthlyIncome = st.number_input('MonthlyIncome', value=0)
    StockOptionLevel = st.number_input('StockOptionLevel', value=0)
    TotalWorkingYears = st.number_input('TotalWorkingYears', value=0)
    YearsAtCompany = st.number_input('YearsAtCompany', value=0)
    YearsInCurrentRole = st.number_input('YearsInCurrentRole', value=0)
    YearsWithCurrManager = st.number_input('YearsWithCurrManager', value=0)

    return age, OverTime, JobLevel, MaritalStatus, MonthlyIncome, StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager

    
# Design the visualisation
def charts():
        
            tab1, tab2, tab3, tab4 = st.tabs(['Bar Chart', 'Line Chart', "2D Scatter Plot", "3D Scatter Plot"])
            with tab1: # bar chart    
                st.bar_chart(df, x=a, y=[b, c], color=['#FF0000', '#0000FF'])  
                
            with tab2: # line chart
                st.line_chart(df, x=x, y=[y, z], color=["#FF0000", "#0000FF"])

            
            with tab3: # 2D scatter plot
                import altair as alt
                #st.scatter_chart(df, x=x, y=[y, z], size=z)
                ch = (alt.Chart(df).mark_circle().encode(x=x, y=y, size=z, color=z, tooltip=[x, y, z]))
                st.altair_chart(ch, use_container_width=True)                                
            
            with tab4: # 3D scatter plot 
                pio.templates.default = 'plotly'
                fig2 = px.scatter_3d(df, x=x, y=z, z=y, size=y, color=x)
                st.plotly_chart(fig2, theme='streamlit', use_container_width=True)            
                fig2.write_image("./media/MDskat.png")
                
def predict_attrition(age, OverTime, JobLevel, MaritalStatus, MonthlyIncome, StockOptionLevel,
                      TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, df):
    # Extract relevant features from the DataFrame
    feature_columns = ['Age', 'OverTime', 'JobLevel', 'MaritalStatus', 'MonthlyIncome',
                       'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany',
                       'YearsInCurrentRole', 'YearsWithCurrManager']

    # Copy relevant features and target variable to a new DataFrame
    features_and_target = df[['Attrition'] + feature_columns].copy()

    # Separate features and target variable
    y1 = features_and_target.loc[:, 'Attrition'].values
    X = features_and_target.drop(columns=['Attrition']).values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y1, test_size=0.2, random_state=141)

    # Initialize the classifier
    params = {'max_depth': 5}
    classifier = DecisionTreeClassifier(**params)

    # Fit the classifier on the training data
    classifier.fit(X_train, y_train)

    # Create a 2D array with input values for prediction
    input_data = np.array([age, OverTime, JobLevel, MaritalStatus, MonthlyIncome, StockOptionLevel,
                           TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager])

    # Reshape to 2D array
    input_data_2d = input_data.reshape(1, -1)

    # Make a prediction
    prediction = classifier.predict(input_data_2d)

    return prediction


    

def cluster(df, y):
    y1 = df.loc[:, y].values
    X = df.drop(columns=[y]).values
    mean_value = np.mean(y1)
    y_class = np.where(y1 > mean_value, 1, 0)
    class0 = X[y_class == 0]
    class1 = X[y_class == 1]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y1, test_size=0.2, random_state=141)
    params = {'max_depth': 5}
    classifier = DecisionTreeClassifier(**params)
    classifier.fit(X_train, y_train)
    gr_data = tree.export_graphviz(classifier, out_file=None, 
                               feature_names=df.drop(columns=[y]).columns,
                               class_names=np.unique(y1).astype(str),
                               filled=True, rounded=True, 
                               special_characters=True)  
    dtree = graphviz.Source(gr_data)
    
    # Convert Graphviz chart to PNG image
    image_data = dtree.pipe(format='png')
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    
    # Streamlit app
    st.title("Decision Tree Visualization")

    # Display the decision tree as an image
    st.image(f"data:image/png;base64,{encoded_image}")
    
# Main 
tab = ''
# tab = '../data/shopping-data.csv'

try:    
    tab = st.file_uploader("Choose a file with datatype .csv)")
    if tab is not None:
        df = readTabData(tab)
        #st.dataframe(df, use_container_width=True)
except:
    pass   
st.success(" Select the attributes of interest")
    
eda(df)
y = viz1(df)
if st.button(":green[See decision tree]"):
            st.subheader("Classification")
            st.write('Classification tree')
            container = st.container()
            cluster(df, y)
x, y, z = viz2(df)
    
if st.button(":green[Explore]"):
            st.subheader("Explore the Data in Diagrams")
            st.write('Click on tabs to explore')
            container = st.container()
            charts()
    
age, OverTime, JobLevel, MaritalStatus, MonthlyIncome, StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager = viz3()
if st.button(":green[Predict]"):
            container = st.container()
            answer = predict_attrition(age, OverTime, JobLevel, MaritalStatus, MonthlyIncome, StockOptionLevel, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager, df)
if answer == 1:
    st.write("Attrition: Yes")
else:
    st.write("Attrition: No")
            
        

    

