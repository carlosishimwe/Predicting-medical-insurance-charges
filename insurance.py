import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pandas_profiling import ProfileReport
from pycaret.regression import load_model, predict_model

import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

#Machine learning
model = load_model('insurance_kaggle')
def predict(model,input_df):
	predictions_df = predict_model(estimator=model,data=input_df)
	predictions = predictions_df['Label'][0]
	return predictions
st.title('Insurance charges prediction app')
st.sidebar.write('Option')

image = Image.open('20356382.jpg')
st.image(image,use_column_width=True)

#option of the sidebar 
option = st.sidebar.selectbox("Choose your analysis option", ('Dataset',  'Exploratory Data analysis ','Predict your insurance charges price'), 2)

#Loading the dataframe 
def load_df():
	df = pd.read_csv("insurance.csv")  # read a CSV file inside the 'data" folder next to 'app.py'
	return df

insurance = load_df()

st.sidebar.image('185893.jpg')

if option == 'Dataset' :
	st.write(insurance)
	st.subheader('The dataset is from Kaggle')
	st.subheader('Columns :')
	st.text('age: age of primary beneficiary')
	st.text('sex: insurance contractor gender, female, male')
	st.text('bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9')
	st.text('children: Number of children covered by health insurance / Number of dependents')
	st.text('smoker: Smoking')
	st.text('region: In the United states , northeast, southeast, southwest, northwest.')
	st.text('charges: Individual medical costs billed by health insurance')
	
#if option == 'Exploratory Data analysis' :
	#df = pd.read_csv("insurance.csv")
	#load_state = st.text('Loading..')
	#load_state.text('Chargement terminé!')
	#profile = ProfileReport(df)
	#st_profile_report(profile)


if option == 'Predict your insurance charges price' :
	age = st.number_input('What is your Age ?',min_value=1,max_value=100,value=20)
	sex = st.selectbox('Sex',['male','female'])
	bmi = st.number_input('What is your bmi ?',min_value=10, max_value=50, value= 25)
	children = st.slider('Children',0, 10)
	if st.checkbox('Smoker'):
		smoker = 'yes'
	else : 
		smoker = 'no'
	region = st.selectbox('Region',['northwest','southwest','northeast','southeast'])

	output=""

	input_dict = {'age': age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region': region}
	input_df = pd.DataFrame([input_dict])

	if st.button("Predict"):
		output = predict(model=model, input_df=input_df)
		output =  '$' + str(round(output))

	st.success(f'Based on the information you provided your insurance charges would be :  {output}')






