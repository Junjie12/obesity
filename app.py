import joblib
model = joblib.load('trained_obesitymodel.pkl')


import streamlit as st
import pandas as pd

st.write("""
# Obesity Prediction App
This app predicts the **Obesity** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():

       # Categorical Features (One-Hot Encoding)
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Gender = 1 if Gender == 'Female' else 0  # 0: Male, 1: Female
    
    # Continuous Features
    Age = st.sidebar.slider('Age', 1, 120, 25)

    Height = st.sidebar.slider('Height (in meters)', 0.5, 2.5, 1.7)

    Weight = st.sidebar.slider('Weight (in kg)', 10, 200, 70)

 

    family_history = st.sidebar.selectbox('Has a family member suffered or suffers from overweight?', ['Yes', 'No'])
    family_history = 1 if family_history == 'Yes' else 0

    FAVC = st.sidebar.selectbox('Do you eat high caloric food frequently?', ['Yes', 'No'])
    FAVC = 1 if FAVC == 'Yes' else 0


    FCVC = st.sidebar.slider('How often do you consume vegetables?', 1, 5, 3)

    
    NCP = st.sidebar.slider('How many meals do you eat per day?', 1, 7, 3)


    SMOKE = st.sidebar.selectbox('Do you smoke?', ['Yes', 'No'])
    SMOKE = 1 if SMOKE == 'Yes' else 0

    CH2O = st.sidebar.slider('How much water do you drink daily? (in liters)', 0.0, 5.0, 2.0)

    SCC = st.sidebar.selectbox('Do you monitor the calories you eat daily?', ['Yes', 'No'])
    SCC = 1 if SCC == 'Yes' else 0

    FAF = st.sidebar.selectbox('How often do you have physical activity?', ['Never', 'Rarely', 'Frequently'])
    FAF = {'Never': 0, 'Rarely': 1, 'Frequently': 2}[FAF]

    
    TUE = st.sidebar.slider('How many hours do you exercise per week?', 0, 20, 5)





    CAEC = st.sidebar.selectbox('Do you eat any food between meals?', ['No', 'Sometimes', 'Frequently', 'Always'])
    CAEC_no = 1 if CAEC == 'No' else 0
    CAEC_sometimes = 1 if CAEC == 'Sometimes' else 0
    CAEC_frequently = 1 if CAEC == 'Frequently' else 0
    CAEC_always = 1 if CAEC == 'Always' else 0


    CALC = st.sidebar.selectbox('How often do you drink alcohol?', ['No', 'Sometimes', 'Frequently', 'Always'])
    CALC_no = 1 if CALC == 'No' else 0
    CALC_sometimes = 1 if CALC == 'Sometimes' else 0
    CALC_frequently = 1 if CALC == 'Frequently' else 0
    CALC_always = 1 if CALC == 'Always' else 0

    
    MTRANS = st.sidebar.selectbox('Which transportation do you usually use?', ['Walking', 'Bike', 'Motorbike', 'Public Transport', 'Automobile'])
    MTRANS_walking = 1 if MTRANS == 'Walking' else 0
    MTRANS_bike = 1 if MTRANS == 'Bike' else 0
    MTRANS_motorbike = 1 if MTRANS == 'Motorbike' else 0
    MTRANS_public_transport = 1 if MTRANS == 'Public Transport' else 0
    MTRANS_automobile = 1 if MTRANS == 'Automobile' else 0


    data = {
        'Gender': Gender,  
        'Age': Age,        
        'Height': Height,  
        'Weight': Weight,  
        'family_history': family_history,
        'FAVC': FAVC,      
        'FCVC': FCVC,      
        'NCP': NCP,        
        'SMOKE': SMOKE, 
        'CH2O': CH2O,  
        'SCC': SCC,      
        'FAF': FAF,       
        'TUE': TUE,        
        'CAEC_Always': CAEC_always,
        'CAEC_Frequently': CAEC_frequently,
        'CAEC_Sometimes': CAEC_sometimes,
        'CAEC_no': CAEC_no,
        'CALC_Always': CALC_always,
        'CALC_Frequently': CALC_frequently,
        'CALC_Sometimes': CALC_sometimes,
        'CALC_no': CALC_no,
        'MTRANS_Automobile': MTRANS_automobile,
        'MTRANS_Bike': MTRANS_bike,
        'MTRANS_Motorbike': MTRANS_motorbike,
        'MTRANS_Public_Transportation': MTRANS_public_transport,
        'MTRANS_Walking': MTRANS_walking
    }
    
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()


st.subheader('User Input parameters')
st.write(df)


prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

# Update the class labels to reflect your obesity categories
class_labels = {
    0: 'Insufficient_Weight', 
    1: 'Normal_Weight', 
    2: 'Obesity_Type_I', 
    3: 'Obesity_Type_II', 
    4: 'Obesity_Type_III', 
    5: 'Overweight_Level_I', 
    6: 'Overweight_Level_II'
}

st.subheader('Class labels')
st.write(class_labels)

# Prediction result
st.subheader('Prediction')
st.write(class_labels[prediction[0]])

# Prediction probabilities
st.subheader('Prediction Probability')
st.write(prediction_proba)
