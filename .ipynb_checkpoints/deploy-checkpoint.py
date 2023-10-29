import streamlit as st 
import numpy as np 
import joblib
# Load the saved model
model_filename = 'polynomial_regression_model2.pkl' 
loaded_model = joblib.load(model_filename)

# Streamlit UI
st.title('Polynomial Regression Model Deployment')

# User input for X values
st.header('Enter X values for prediction:')

x_input = st.text_input('X Values (comma-separated)', '6,2.5,3.4,4,5.2')


# Convert user input to a NumPy array
try:
    x_input = np.array([float(x.strip()) for x in x_input.split(',')]).reshape(-1, 1)

except ValueError:
    st.error('Invalid input. Please enter comma-separated numeric values.') 
    x_input = None

# Predict and display results when the "Predict" button is clicked
if st.button('Predict'):
    if x_input is not None:
        st.write('Predicted Y values:')

y_pred = loaded_model.predict(x_input)

for i, pred in enumerate(y_pred): 
    st.write(f'X={x_input[i][0]}, Predicted Y={pred}')