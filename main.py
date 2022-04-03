import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn




original_title = '<p style=" color:Blue; font-size: 40px;">This application detected the presence of heart disease in the patient </p>'
st.markdown(original_title, unsafe_allow_html=True)
img = mpimg.imread('heart.jpg')
#img = cv2.imread('heart.jpg', 1)
#image = np.array([img])
st.image(img, channels="BGR")
st.write("- 0 = no disease")
st.write("- 1 = disease")
st.title("")

#collecter
st.sidebar.header("Attributs : ")
def patient_attributs_entree():

    age=st.sidebar.slider('age',10,100,10)
    sex=st.sidebar.selectbox('sex',(0,1))
    cp=st.sidebar.selectbox('chest pain type',(0,1,2,3))
    trestbps=st.sidebar.slider('resting blood pressure',50,300,20)
    chol=st.sidebar.slider('serum cholestoral in mg/dl',50,1000,50)
    fbs=st.sidebar.selectbox('(fasting blood sugar &gt; 120 mg/dl)',(0,1))
    restecg=st.sidebar.selectbox('electrocardiographic results',(0,1,2))
    thalach=st.sidebar.slider('maximum heart rate achieved',0,700,200)
    exang=st.sidebar.selectbox('exercise induced angina',(1,0))
    oldpeak=st.sidebar.slider('ST depression induced by exercise relative to rest',0.0,20.0,1.0)
    slope=st.sidebar.selectbox('slope',(0,1,2)),
    ca = st.sidebar.selectbox('ca', (0, 1, 2,3,4)),
    thal = st.sidebar.selectbox('thal', (0, 1, 2, 3)),


    data = {
        'age':age,
        'sex':sex,
        'cp':cp,
        'trestbps': trestbps,
        'chol':chol,
        'fbs':fbs,
        'restecg':restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope':slope,
        'ca':ca,
        'thal':thal

    }
    attribut_patient = pd.DataFrame(data,index=[0])
    return attribut_patient

input_df = patient_attributs_entree()
df = pd.read_csv('heart.csv')
patient_input= df.drop(columns=['target'])
donne_entree=pd.concat([input_df,patient_input],axis=0)





donne_entree=donne_entree[:1]
st.subheader('Entred attributs ')
st.write(donne_entree)


#importé le modèle
load_model=pickle.load(open('heart.pkl','rb'))


prevision=load_model.predict(donne_entree)

st.subheader('Résultat de la prévision')
st.write(prevision)
st.write(df)










