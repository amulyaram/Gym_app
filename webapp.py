#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


import pickle


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


model = pickle.load(open("Random_model.pkl","rb"))


# In[6]:


st.set_page_config(page_title="GYM Application")


# In[7]:


#prediction function
def predict_status(age, height, weight):
    input_data = np.asarray([age, height, weight])
    input_data = input_data.reshape(1,-1)
    prediction = model.predict(input_data)
    return prediction[0]


# In[8]:


def main():

    # titling your page
    st.title("Fitness Prediction App")
    st.write("A quick Machine Learning model that predicts the health status based on the given dataset")


# In[12]:


#getting the input
age=st.text_input("Enter your Age")
height=st.text_input("Enter your Height in Feet")
weight=st.text_input("Enter your Weight in Pounds")


# In[13]:


#predict value
diagnosis = ""


# In[14]:


if st.button("Predict"):
       diagnosis = predict_status(age, height, weight)
       if diagnosis=="Underweight":
           st.info("You're Under Weight")
           st.markdown("![You're like this!](https://i.gifer.com/L6m.gif)")
       elif diagnosis=="Healthy":
           st.success("You're Healthy")
           st.markdown("![keep up your health dude!](https://i.gifer.com/lf.gif)")
       elif diagnosis=="Overweight":
           st.warning("You're Over Weight")
           st.markdown("![Go Excercise!](https://i.gifer.com/8TK.gif)")
       elif diagnosis=="Obese":
           st.error("Obesity!")
           st.markdown("![You need HELP!](https://i.gifer.com/5Waz.gif)")
       else:
           st.error("Extremely Obese")
           st.markdown("![You need HELP!](https://i.gifer.com/VYAM.gif)")


# In[16]:


st.subheader("what next? Take Action Towards Better Health")
st.write("🙋🏼‍♂️ Maintaining a healthy weight is important for your heart health")
st.write("🙋🏼‍♂️ Don't be like Spongebob or Patrik")
st.write("Maintain a Healthy Weight: [ check right now!](https://www.nhlbi.nih.gov/heart-truth/maintain-a-healthy-weight)")


# In[18]:


st.write("## Thank you for Visiting \nProject by Amulya Ch")
st.markdown("<h1 style='text-align: right; color: blue; font-size: small;'><a href='https://github.com/Nikhil-Jagtap619/gym_app'>Looking for Source Code?</a></h1>", unsafe_allow_html=True)


# In[19]:


if __name__=="__main__":
    main()


# In[ ]:




