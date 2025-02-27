import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Use joblib for better compatibility

# Load the saved classification model
def load_model():
    try:
        model = joblib.load("classifier.pkl")
        st.success("Model loaded successfully.")
    except FileNotFoundError:
        st.error("No saved model found. Please train and save a model first.")
        model = None
    return model

# Function to preprocess user input
def preprocess_input(user_input):
    # Example: Convert categorical inputs to numerical (Modify based on your dataset)
    user_input['Gender'] = 1 if user_input['Gender'] == "Male" else 0
    user_input['BMI'] = float(user_input['BMI'])
    user_input['Sleep Duration'] = float(user_input['Sleep Duration'])
    user_input['Stress Level'] = float(user_input['Stress Level'])
    
    # Convert dictionary to NumPy array
    input_array = np.array(list(user_input.values())).reshape(1, -1)
    
    return input_array

# Streamlit UI
def main():
    st.title("Sleep Disorder Classification App")
    st.write("Enter your details, and the model will predict your sleep disorder status.")

    model = load_model()  # Load model on app startup

    if model is not None:
        # User Input Form
        st.write("### Enter Your Information")
        user_input = {
            "Age": st.number_input("Age", min_value=10, max_value=100, value=25),
            "Gender": st.selectbox("Gender", ["Male", "Female"]),
            "BMI": st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0),
            "Sleep Duration": st.number_input("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.0),
            "Stress Level": st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5),
        }

        if st.button("Predict"):
            # Preprocess input
            processed_input = preprocess_input(user_input)

            # Make prediction
            prediction = model.predict(processed_input)

            # Display result
            st.write("### Prediction Result")
            st.write(f"The predicted sleep disorder status is: **{prediction[0]}**")

            # Convert user input to DataFrame and allow download
            df = pd.DataFrame([user_input])
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Processed Data as CSV",
                data=csv,
                file_name="processed_input.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
