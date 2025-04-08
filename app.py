import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image

# 1. Load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# 2. Group the data by Outcome to get a sense of the distribution
mean_df = df.groupby('Outcome').mean()

# 3. Prepare input features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Scale the input variables using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# 6. Create an SVM model with a linear kernel and train it
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# 7. Make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# 8. Calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(y_train, train_y_pred)
test_acc = accuracy_score(y_test, test_y_pred)

# 9. Define the Streamlit app
def app():
    # Display an image (ensure 'img.jpeg' is in the correct path)
    try:
        img = Image.open("img.jpeg")
        img = img.resize((200, 200))
        st.image(img, caption="Diabetes Image", width=200)
    except FileNotFoundError:
        st.error("Image file not found. Please check the file path.")

    st.title('Diabetes Prediction')

    # Create the input form for the user to input new data via the sidebar
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
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    # Scale the new input data using the previously fitted scaler
    reshaped_input_data_scaled = scaler.transform(reshaped_input_data)
    
    # Make prediction using the trained SVM model
    prediction = model.predict(reshaped_input_data_scaled)

    # Display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction[0] == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # Display summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(df.describe())

    st.header('Distribution by Outcome (Mean Values)')
    st.write(mean_df)

    # Display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

# 10. Run the Streamlit app
if __name__ == '__main__':
    app()
