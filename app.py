import gradio as gr
import pickle


# Load the trained Logistic Regression model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the prediction function
def predict_admission(GRE, TOEFL, Rating, SOP, LOR, CGPA, Research):
    input_features = [[GRE, TOEFL, Rating, SOP, LOR, CGPA, Research]]
    prediction = model.predict(input_features)
    return "✅ Admitted" if prediction[0] == 1 else "❌ Not Admitted"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_admission,
    inputs=[
        gr.Number(label="GRE Score"),
        gr.Number(label="TOEFL Score"),
        gr.Number(label="University Rating"),
        gr.Slider(1.0, 5.0, step=0.5, label="SOP Score"),
        gr.Slider(1.0, 5.0, step=0.5, label="LOR Score"),
        gr.Number(label="CGPA"),
        gr.Radio([0, 1], label="Research (0 = No, 1 = Yes)")
    ],
    outputs=gr.Textbox(label="Admission Prediction"),
    title="🎓 University Admission Predictor",
    description="Predict whether a student will get admitted to a university based on academic metrics.",
    theme="soft"
)

# Launch the app
interface.launch()
