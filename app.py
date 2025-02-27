import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

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
    
    tab1, tab2 = st.tabs(["Prediction", "Visualizations"])
    
    with tab1:
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
    
    with tab2:
        st.write("### Data Visualizations")
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

        # Plotly Bar Plot - Age vs Sleep Disorder by Gender
        fig = px.bar(df, x='Sleep Disorder', y='Age', color='Gender', title="Age vs. Sleep Disorder by Gender")
        st.plotly_chart(fig)

        # Plotly Scatter Plot - Quality of Sleep vs Stress Level
        fig = px.scatter(df, x='Quality of Sleep', y='Stress Level', title="Quality of Sleep vs. Stress Level", trendline="ols")
        st.plotly_chart(fig)

        # Plotly Bar Plot - BMI Category vs Stress Level
        fig = px.bar(df, x='BMI Category', y='Stress Level', title="BMI Category vs. Stress Level")
        st.plotly_chart(fig)

        # Plotly Scatter Plot - Sleep Duration vs Stress Level
        fig = px.scatter(df, x='Sleep Duration', y='Stress Level', title="Sleep Duration vs. Stress Level", trendline="ols")
        st.plotly_chart(fig)

        # Crosstab Visualizations with Plotly
        gender_vs_occupation = pd.crosstab(df['Gender'], df['Occupation']).reset_index()
        fig = px.bar(gender_vs_occupation, x='Gender', y=gender_vs_occupation.columns[1:], title="Gender vs. Occupation")
        st.plotly_chart(fig)

        sleep_disorder_vs_occupation = pd.crosstab(df['Sleep Disorder'], df['Occupation']).reset_index()
        fig = px.bar(sleep_disorder_vs_occupation, x='Sleep Disorder', y=sleep_disorder_vs_occupation.columns[1:], title="Sleep Disorder vs. Occupation")
        st.plotly_chart(fig)

        bmi_vs_occupation = pd.crosstab(df['BMI Category'], df['Occupation']).reset_index()
        fig = px.bar(bmi_vs_occupation, x='BMI Category', y=bmi_vs_occupation.columns[1:], title="BMI Category vs. Occupation")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
