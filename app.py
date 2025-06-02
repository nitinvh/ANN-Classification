import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

st.title('Customer Churn Prediction')

model = tensorflow.keras.models.load_model('model.h5')


with open('onehot_encoder_geo.pkl','rb') as file:
    ohe_pkl = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    le_pkl = pickle.load(file)

with open('scaler.pkl','rb') as file:
    sc_pkl = pickle.load(file)

#user input
creditScore = st.slider('Credit Score',100,850)
geography = st.selectbox('Geography',ohe_pkl.categories_[0])
gender = st.selectbox('Gender',le_pkl.classes_)
age = st.slider('Age',18,95)
balance = st.number_input('Balance')
tenure = st.number_input('Tenure')
no_of_products = st.number_input('No of Products')
hasCrCard = st.selectbox('Has Credit Card',[0,1])
isActiveMember = st.selectbox('Is Active Member',[0,1])
estimatedSalary = st.number_input('Estimated Salary')

input_data ={
    'CreditScore':creditScore,
    'Geography':geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts':no_of_products,
    'HasCrCard':hasCrCard,
    'IsActiveMember':isActiveMember,
    'EstimatedSalary': estimatedSalary
}

input_df = pd.DataFrame([input_data],columns=input_data.keys())

geo_encoder = ohe_pkl.transform([input_df['Geography']]).toarray()
geo_encoder = pd.DataFrame(geo_encoder,columns=ohe_pkl.get_feature_names_out())

input_df['Gender'] = le_pkl.transform(input_df['Gender'])

input_df.drop(['Geography'],axis=1,inplace=True)

input_df = pd.concat([input_df,geo_encoder],axis=1)

input_df_scaled = sc_pkl.transform(input_df)

pred = model.predict(input_df_scaled)

st.write(f"Churn Probability {pred[0][0]}")

if pred[0][0] < 0.5:
    st.write('The customer is not likely to churn.')
else:
    st.write('The customer is likely to churn.')

