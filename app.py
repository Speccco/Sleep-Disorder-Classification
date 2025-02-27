import streamlit as st
import numpy as np
import pickle

# Load the saved classification model
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("Model loaded successfully.")
    except FileNotFoundError:
        st.error("No saved model found. Please train and save a model first.")
        model = None
    return model

# Streamlit UI
def main():
    st.title("Classification Model Prediction App")
    st.write("Enter input values and get a prediction from the saved model.")
    
    model = load_model()
    
    if model is not None:
        st.write("### Enter Feature Values")
        
        num_features = len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 4
        input_values = []
        
        for i in range(num_features):
            value = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f")
            input_values.append(value)
        
        if st.button("Predict"):
            input_array = np.array([input_values]).reshape(1, -1)
            prediction = model.predict(input_array)
            
            st.write("### Prediction Result")
            st.write(f"The predicted class is: **{prediction[0]}**")

if __name__ == "__main__":
    main()
