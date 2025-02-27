import streamlit as st
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        
        fig, ax = plt.subplots()
        sns.barplot(data=df, y='Age', x='Sleep Disorder', hue='Gender', ax=ax)
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.regplot(data=df, x='Quality of Sleep', y='Stress Level', ax=ax)
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='BMI Category', y='Stress Level', ax=ax)
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sns.regplot(data=df, x='Sleep Duration', y='Stress Level', ax=ax)
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        pd.crosstab(df['Gender'], df['Occupation']).plot(kind='bar', ax=ax)
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
        plt.tight_layout()
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        pd.crosstab(df['Sleep Disorder'], df['Occupation']).plot(kind='bar', ax=ax)
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
        plt.tight_layout()
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        pd.crosstab(df['BMI Category'], df['Occupation']).plot(kind='bar', ax=ax)
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
