"""
Tips Prediction Web Application
Gradio Interface for Restaurant Tip Prediction Model
"""

import gradio as gr
import pickle
import numpy as np
import pandas as pd

# Load the trained Random Forest model
print("Loading trained model...")
try:
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️ Model file not found. Please run the Jupyter notebook first to train the model.")
    model = None


def predict_tip(total_bill, size, sex, smoker, day, time):
    """
    Predict tip amount based on input features.

    Parameters:
    -----------
    total_bill : float
        Total bill amount in dollars
    size : int
        Number of people in the party
    sex : str
        Gender of the bill payer ('Male' or 'Female')
    smoker : str
        Whether party includes smokers ('Yes' or 'No')
    day : str
        Day of the week
    time : str
        Meal time ('Lunch' or 'Dinner')

    Returns:
    --------
    tuple : (prediction_text, details_text)
        Formatted prediction results
    """

    if model is None:
        return "⚠️ Model not loaded. Please train the model first.", ""

    # Input validation
    if total_bill <= 0:
        return "⚠️ Error: Total bill must be greater than 0", ""

    if size < 1 or size > 10:
        return "⚠️ Error: Party size must be between 1 and 10", ""

    try:
        # Encode categorical variables (same encoding as training)
        sex_encoded = 1 if sex == 'Male' else 0
        smoker_encoded = 1 if smoker == 'Yes' else 0

        day_mapping = {'Fri': 0, 'Sat': 1, 'Sun': 2, 'Thur': 3}
        day_encoded = day_mapping.get(day, 0)

        time_encoded = 0 if time == 'Dinner' else 1

        # Create feature array
        features = np.array([[total_bill, size, sex_encoded, smoker_encoded,
                              day_encoded, time_encoded]])

        # Make prediction
        tip_prediction = model.predict(features)[0]

        # Ensure non-negative prediction
        tip_prediction = max(0, tip_prediction)

        # Calculate tip percentage
        tip_percentage = (tip_prediction / total_bill) * 100

        # Total with tip
        total_with_tip = total_bill + tip_prediction

        # Format main prediction
        prediction_text = f"""
        🎯 **Predicted Tip Amount: ${tip_prediction:.2f}**

        📊 **Breakdown:**
        - Original Bill: ${total_bill:.2f}
        - Predicted Tip: ${tip_prediction:.2f} ({tip_percentage:.1f}%)
        - Total with Tip: ${total_with_tip:.2f}
        """

        # Format details
        details_text = f"""
        **Input Details:**
        - Party Size: {size} {'person' if size == 1 else 'people'}
        - Day: {day}
        - Time: {time}
        - Customer: {sex}
        - Smoker: {smoker}

        **Model Information:**
        - Algorithm: Linear Regression
        - Features Used: 6 (bill, size, gender, smoker, day, time)
        - Training Samples: 244 restaurant transactions

        **Note:** This prediction is based on historical restaurant tip data. 
        Actual tips may vary based on service quality and individual preferences.
        """

        return prediction_text, details_text

    except Exception as e:
        return f"⚠️ Error making prediction: {str(e)}", ""


# Create Gradio Interface
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Restaurant Tip Predictor") as interface:
        # Header
        gr.Markdown("""
        # 🍽️ Restaurant Tip Predictor
        ### AI-Powered Tip Amount Prediction System

        This application uses machine learning to predict restaurant tip amounts based on 
        transaction characteristics. Enter the details below to get an instant prediction.
        """)

        with gr.Row():
            # Left Column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input Information")

                total_bill = gr.Number(
                    label="Total Bill Amount ($)",
                    value=25.00,
                    minimum=0.01,
                    maximum=200.00,
                    step=0.01,
                    info="Enter the total bill amount before tip"
                )

                size = gr.Slider(
                    label="Party Size",
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    info="Number of people in the party"
                )

                sex = gr.Radio(
                    label="Customer Gender",
                    choices=["Male", "Female"],
                    value="Male",
                    info="Gender of the bill payer"
                )

                smoker = gr.Radio(
                    label="Smoker Status",
                    choices=["No", "Yes"],
                    value="No",
                    info="Is anyone in the party smoking?"
                )

                day = gr.Dropdown(
                    label="Day of Week",
                    choices=["Thur", "Fri", "Sat", "Sun"],
                    value="Sat",
                    info="Select the day of the visit"
                )

                time = gr.Radio(
                    label="Meal Time",
                    choices=["Dinner", "Lunch"],
                    value="Dinner",
                    info="Time of the meal"
                )

                predict_btn = gr.Button("🔮 Predict Tip Amount", variant="primary", size="lg")

            # Right Column - Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Prediction Results")

                prediction_output = gr.Markdown(
                    label="Prediction",
                    value="*Enter details and click 'Predict Tip Amount' to see results*"
                )

                details_output = gr.Markdown(
                    label="Details",
                    value=""
                )

        # Examples
        gr.Markdown("### 💡 Example Scenarios")
        gr.Examples(
            examples=[
                [25.50, 2, "Male", "No", "Sat", "Dinner"],
                [48.27, 4, "Male", "No", "Sat", "Dinner"],
                [15.98, 3, "Female", "No", "Thur", "Lunch"],
                [35.26, 4, "Female", "No", "Sun", "Dinner"],
                [10.34, 3, "Male", "No", "Sun", "Dinner"],
            ],
            inputs=[total_bill, size, sex, smoker, day, time],
            label="Click an example to try it"
        )

        # Footer
        gr.Markdown("""
        ---
        ### 📚 About This Model

        This tip prediction model was trained on 244 restaurant transactions using a 
        **Linear Regression** algorithm. The model achieved:
        - **RMSE**: ~$1.01 (typical prediction error)
        - **R² Score**: ~0.444 (explains 44% of tip variance)
        - **Most Important Feature**: Total bill amount (~70% importance)

        The model uses 6 features to make predictions and was validated using cross-validation 
        techniques to ensure reliability.

        ---
        *Machine Learning Project - Tips Prediction System*  
        *Developed for educational purposes*
        """)

        # Connect button to prediction function
        predict_btn.click(
            fn=predict_tip,
            inputs=[total_bill, size, sex, smoker, day, time],
            outputs=[prediction_output, details_output]
        )

    return interface


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("RESTAURANT TIP PREDICTOR - GRADIO WEB APPLICATION")
    print("=" * 80)

    # Create and launch interface
    interface = create_interface()

    print("\n🚀 Launching web interface...")
    print("=" * 80)

    # Launch with configuration
    interface.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Local access only
        server_port=7860,  # Default Gradio port
        show_error=True,
        quiet=False
    )

    print("\n✓ Interface closed.")
    print("=" * 80)