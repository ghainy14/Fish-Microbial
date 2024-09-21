import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('microbial_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
pigmentation = st.text_input("Pigmentation1", "")
elevation = st.text_input("Elevation1", "")
texture_1 = st.text_input("Texture1", "")
margin = st.text_input("Margin1", "")
shape = st.text_input("Shape1", "")
optical_density = st.text_input("Optical Density1", "")
pigmentation_1 = st.text_input("Pigmentation2", "")
elevation_1 = st.text_input("Elevation2", "")
texture_2 = st.text_input("Texture2", "")
margin_1 = st.text_input("Margin2", "")
shape_1 = st.text_input("Shape2", "")
optical_density_1 = st.text_input("Optical Density2", "")
pigmentation_2 = st.text_input("Pigmentation3", "")
elevation_2 = st.text_input("Elevation3", "")
texture_3 = st.text_input("Texture3", "")
margin_2 = st.text_input("Margin3", "")
shape_2 = st.text_input("Shape3", "")
optical_density_2 = st.text_input("Optical Density3", "")
ph = st.number_input("pH", min_value=0.0)
lipid_oxidation = st.number_input("Lipid Oxidation", min_value=0.0)
moisture_content = st.number_input("Moisture Content", min_value=0.0)
protein = st.number_input("Protein", min_value=0.0)
fat = st.number_input("Fat", min_value=0.0)
ash = st.number_input("Ash", min_value=0.0)

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
    'Pigmentation': [pigmentation1],
    'Elevation': [elevation1],
    'Texture.1': [texture1],
    'Margin': [margin1],
    'Shape': [shape1],
    'Optical Density': [optical_density1],
    'Pigmentation.1': [pigmentation2],
    'Elevation.1': [elevation2],
    'Texture.2': [texture2],
    'Margin.1': [margin2],
    'Shape.1': [shape2],
    'Optical Density.1': [optical_density2],
    'Pigmentation.2': [pigmentation3],
    'Elevation.2': [elevation3],
    'Texture.3': [texture3],
    'Margin.2': [margin3],
    'Shape.2': [shape3],
    'Optical Density.2': [optical_density3],
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


