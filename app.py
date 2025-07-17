import streamlit as st
import numpy as np
from inference import load_model, predict_next_number

st.title("🧠 LSTM Sequence Number Predictor")

# Input sequence
input_values = st.text_input("Enter 3 numbers (comma-separated):", "45,46,47")

if st.button("Predict Next Number"):
    try:
        nums = [int(i.strip()) for i in input_values.split(",")]
        if len(nums) != 3:
            st.error("Enter exactly 3 numbers!")
        else:
            model = load_model()
            result = predict_next_number(model, nums)
            st.success(f"🔮 Predicted next number: {result:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

