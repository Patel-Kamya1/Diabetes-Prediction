import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from PIL import Image
import joblib  # For loading the saved model

# Load the trained model (make sure 'Diabetes.pkl' exists in the directory)
best_model = joblib.load('Diabetes.pkl')

# Load scaler (if you saved it separately during training)
scaler = joblib.load('scaler.pkl')  # Ensure the scaler is saved and loaded properly

# Define the Streamlit app
def app():
    # Display an image (ensure 'img.jpeg' is in the correct path)
    try:
        img = Image.open("img.jpeg")
        img = img.resize((200, 200))
        st.image(img, caption="Diabetes Image", width=200)
    except FileNotFoundError:
        st.error("Image file not found. Please check the file path.")

    st.title('Diabetes Prediction with Pre-trained Model')

    # Sidebar for user input
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Prepare input data for prediction
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data).reshape(1, -1)

    # Scale the input data using the saved scaler
    input_data_scaled = scaler.transform(input_data_nparray)

    # Prediction
    prediction = best_model.predict(input_data_scaled)

    # Display the prediction
    st.write('Based on the input features, the model predicts:')
    if prediction[0] == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # Load dataset for summary statistics
    df = pd.read_csv('Diabetes.csv')
    mean_df = df.groupby('Outcome').mean()

    # Display dataset information
    st.header('Dataset Summary')
    st.write(df.describe())

    st.header('Distribution by Outcome (Mean Values)')
    st.write(mean_df)

    # Add graphical comparison of outcomes
    st.header('Graphical Representation')

    # Outcome distribution bar chart
    outcome_count = df['Outcome'].value_counts()
    st.bar_chart(outcome_count)

    # Display graphical relationship between Age and Glucose levels for each Outcome
    st.subheader('Age vs Glucose Levels by Outcome')
    plot_df = df[['Age', 'Glucose', 'Outcome']].groupby('Outcome').mean()
    st.line_chart(plot_df)

# Run the Streamlit app
if __name__ == '__main__':
    app()