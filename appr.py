import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('microbial_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_model():
    with open('scalers.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Streamlit app layout
st.title("Microbial Organisms Multi-Label Prediction App")

# Example: Input fields (customize these based on your feature space)
st.header("Input Predicting Variables")
# Collecting input values for all feature columns
fish_sample = st.text_input("Fish Sample", "")
colour = st.text_input("Colour", "")
odour = st.text_input("Odour", "")
texture = st.text_input("Texture", "")
flavour = st.text_input("Flavour", "")
appearance = st.text_input("Appearance", "")
insect_invasion = st.text_input("Insect Invasion", "")
overall_acceptability = st.text_input("Overall Acceptability", "")
market = st.text_input("Market", "")
state = st.text_input("State", "")
tba = st.text_input("TBA", "")
pbc = st.text_input("PBC", "")
tfc = st.text_input("TFC", "")
pigmentation1 = st.text_input("Pigmentation1", "")
elevation1 = st.text_input("Elevation1", "")
texture1 = st.text_input("Texture1", "")
margin1 = st.text_input("Margin1", "")
shape1 = st.text_input("Shape1", "")
optical_density1 = st.text_input("Optical Density1", "")
pigmentation2 = st.text_input("Pigmentation2", "")
elevation2 = st.text_input("Elevation2", "")
texture2 = st.text_input("Texture2", "")
margin2 = st.text_input("Margin2", "")
shape2 = st.text_input("Shape2", "")
optical_density2 = st.text_input("Optical Density2", "")
pigmentation3 = st.text_input("Pigmentation3", "")
elevation3 = st.text_input("Elevation3", "")
texture3 = st.text_input("Texture3", "")
margin3 = st.text_input("Margin3", "")
shape3 = st.text_input("Shape3", "")
optical_density3 = st.text_input("Optical Density3", "")
ph = st.number_input("pH", min_value=0.0)
lipid_oxidation = st.number_input("Lipid Oxidation (MDA) mg/L", min_value=0.0)
moisture_content = st.number_input("Moisture Content (%)", min_value=0.0)
protein = st.number_input("Protein (%)", min_value=0.0)
fat = st.number_input("Fat (%)", min_value=0.0)
ash = st.number_input("Ash (%)", min_value=0.0)

# Combine input data into a DataFrame
input_data = pd.DataFrame({
    'Fish sample': [fish_sample],
    'Colour': [colour],
    'Odour': [odour],
    'Texture': [texture],
    'Flavour': [flavour],
    'Appearance': [appearance],
    'Insect Invasion': [insect_invasion],
    'Overall Acceptability': [overall_acceptability],
    'Market': [market],
    'State': [state],
    'TBA': [tba],
    'PBC': [pbc],
    'TFC': [tfc],
    'Pigmentation1': [pigmentation1],
    'Elevation1': [elevation1],
    'Texture1': [texture1],
    'Margin1': [margin1],
    'Shape1': [shape1],
    'Optical Density1': [optical_density1],
    'Pigmentation2': [pigmentation2],
    'Elevation2': [elevation2],
    'Texture2': [texture2],
    'Margin2': [margin2],
    'Shape2': [shape2],
    'Optical Density2': [optical_density2],
    'Pigmentation3': [pigmentation3],
    'Elevation3': [elevation3],
    'Texture3': [texture3],
    'Margin3': [margin3],
    'Shape3': [shape3],
    'Optical Density3': [optical_density3],
    'pH': [ph],
    'Lipid oxidation': [lipid_oxidation],
    'Moisture Content': [moisture_content],
    'Protein': [protein],
    'Fat': [fat],
    'Ash': [ash]
})

# Button to trigger prediction
if st.button("Predict"):
    try:
        # Ensure the input is preprocessed the same way as during training
        # (This step may include scaling or encoding depending on how the model was trained)
        
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        
        # Assuming your model returns labels in a human-readable format
        st.subheader("Predicted Microbial Organisms:")
        st.write(prediction)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.write("This application uses a multi-label classification model to predict microbial organisms based on input features.")
