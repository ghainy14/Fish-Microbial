import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib
model = joblib.load('microbial_model.pkl')

# Load the trained model
#@st.cache(allow_output_mutation=True)
#def load_model():
 #   with open('microbial_model.pkl', 'rb') as file:
  #      model = pickle.load(file)
   # return model

# Load the model, scaler, and encoder
#model = load_model()
#scaler = load_scaler()
#encoder = load_encoder()

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
pigmentation = st.text_input("Pigmentation", "")
elevation = st.text_input("Elevation", "")
texture_1 = st.text_input("Texture.1", "")
margin = st.text_input("Margin", "")
shape = st.text_input("Shape", "")
optical_density = st.text_input("Optical Density", "")
pigmentation_1 = st.text_input("Pigmentation.1", "")
elevation_1 = st.text_input("Elevation.1", "")
texture_2 = st.text_input("Texture.2", "")
margin_1 = st.text_input("Margin.1", "")
shape_1 = st.text_input("Shape.1", "")
optical_density_1 = st.text_input("Optical Density.1", "")
pigmentation_2 = st.text_input("Pigmentation.2", "")
elevation_2 = st.text_input("Elevation.2", "")
texture_3 = st.text_input("Texture.3", "")
margin_2 = st.text_input("Margin.2", "")
shape_2 = st.text_input("Shape.2", "")
optical_density_2 = st.text_input("Optical Density.2", "")
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
    'Pigmentation': [pigmentation],
    'Elevation': [elevation],
    'Texture.1': [texture_1],
    'Margin': [margin],
    'Shape': [shape],
    'Optical Density': [optical_density],
    'Pigmentation.1': [pigmentation_1],
    'Elevation.1': [elevation_1],
    'Texture.2': [texture_2],
    'Margin.1': [margin_1],
    'Shape.1': [shape_1],
    'Optical Density.1': [optical_density_1],
    'Pigmentation.2': [pigmentation_2],
    'Elevation.2': [elevation_2],
    'Texture.3': [texture_3],
    'Margin.2': [margin_2],
    'Shape.2': [shape_2],
    'Optical Density.2': [optical_density_2],
    'pH': [ph],
    'Lipid oxidation': [lipid_oxidation],
    'Moisture Content': [moisture_content],
    'Protein': [protein],
    'Fat': [fat],
    'Ash': [ash]
})
if st.button("Predict"):
    try:
        # Handle preprocessing here
        # Apply scaling to numerical fields (if necessary)
        numeric_fields = input_data.select_dtypes(include=np.number)
        input_data_scaled = scaler.transform(numeric_fields)
        
        # Apply encoding to categorical fields (LabelEncoder for input fields)
        df_to_encode = input_data.select_dtypes(include='object').astype(str)
        le = LabelEncoder()
        for column in df_to_encode.columns:
            df_to_encode[column] = le.fit_transform(df_to_encode[column])
        
        # Combine the scaled and encoded data
        input_preprocessed = np.hstack([input_data_scaled, df_to_encode])
        
        # Convert preprocessed input back into a DataFrame
        input_preprocessed_df = pd.DataFrame(input_preprocessed, columns=input_data.columns)

        # Make predictions with the preprocessed data
        prediction = model.predict(input_preprocessed_df)

        # Decode the multilabel prediction (OneHotEncoder reverse transformation)
        predicted_labels = encoder.inverse_transform(prediction)

        # Display the prediction
        st.subheader("Predicted Microbial Organisms:")
        st.write(predicted_labels)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.write("This application uses a multi-label classification model to predict microbial organisms based on input features.")


