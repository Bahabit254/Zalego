from tkinter import Button
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib
import sklearn


def main():
    st.title('Heart disease prediction using Random Forest')
    
    filename='rfr_model.pkl'
    loaded_model=joblib.load(filename)
    
    
    
    col_1, col_2=st.columns(2)
    with col_1:
        Sex=st.selectbox('Sex:',['Female','Male'])
        if Sex=='Female':
            Sex=0
        else:
            Sex=1
        
        Exercise_angina=st.selectbox('Exercise_angina:',['No','Yes'])
        if Exercise_angina=='No':
            Exercise_angina=0
        else:
            Exercise_angina=1
        
        FBS_over_120=st.selectbox('FBS_over_120:',['No','Yes'])
        if FBS_over_120=='No':
            FBS_over_120=0
        else:
            FBS_over_120=1

        Age=st.number_input('Age:',min_value=1,max_value=100)

        Chest_pain_type=st.slider('Chest_pain_type:',min_value=1,max_value=4)

    with col_2:
        BP=st.number_input('BloodPressure:',min_value=50,max_value=300)

        Cholesterol=st.number_input('Cholesterol:',min_value=100,max_value=700)

        Max_HR=st.number_input('Maximum_HeartRate:',min_value=50,max_value=300)

        ST_depression=st.number_input('ST_depression:',min_value=0.00,max_value=10.00)

        ST_slope=st.number_input('ST_slope:',min_value=0,max_value=10)

        vessels_fluro=st.number_input('vessels_fluro:',min_value=0,max_value=5)

        Thallium=st.number_input('Thallium:',min_value=0,max_value=10)


    input_dict={'Sex':'Sex','Exercise_angina':'Exercise_angina','FBS_over_120':'FBS_over_120','Age':'Age','Chest_pain_type':'Chest_pain_type','BP':'BloodPressure','Cholesterol':'Cholesterol','Max_HR':'Maximum_HeartRate','ST_depression':'ST_depression','ST_slope':'ST_slope','vessels_fluro':'vessels_fluro','Thallium':'Thallium'}
    input_df=pd.DataFrame(input_dict,index=[0])

    Button=st.button('Predict')

    if Button:
        risk= loaded_model.predict(input_df)
        if risk==0:
            st.success('Low risk of heart diesease')
        else:
            st.error('High risk of heart disease')


        precision, recall, f1, acc = st.columns(4)
        st.markdown("""<style>div[data-testid="metric-container"] {background-color: rgba(28, 131, 225, 0.1);border: 1px solid rgba(28, 131, 225, 0.1);padding: 5% 5% 5% 10%;border-radius: 5px;color: rgb(30, 103, 119);overflow-wrap: break-word;} /* breakline for metric text         */div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {overflow-wrap: break-word;white-space: break-spaces;color: green;font-size: 20px;} </style>""", unsafe_allow_html=True)

        with precision:
            st.metric(label="Precision Score", value="84%")
        with recall:
            st.metric(label="Recall Score", value="84%")
    
        with f1:
            st.metric(label="F1 Score", value="84%")
        with acc:
            st.metric(label="Accuracy Score", value="84%")



main()