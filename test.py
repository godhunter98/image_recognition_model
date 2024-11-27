import streamlit as st
import pandas as pd
import numpy as np

# Title of the app
st.title('My First Streamlit App')

# Creating a dataframe
df = pd.DataFrame({
    'First Column': [1, 2, 3, 4],
    'Second Column': [10, 20, 30, 40]
})

# Display the dataframe
st.write("Here's a simple dataframe:")
st.write(df)

# Creating a chart
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C'])

st.line_chart(chart_data)

# Adding an interactive slider
slider_value = st.slider('Select a value', 0, 100, 25)
st.write('Selected value:', slider_value)
