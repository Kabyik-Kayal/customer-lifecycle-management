from flask import Flask, request, render_template, jsonify
import pickle
"""
This module implements a Flask web application for predicting Customer Lifetime Value (CLTV) using a pre-trained model.

Routes:
    /: Renders the home page of the customer lifecycle management application.
    /predict: Handles POST requests to predict customer lifetime value (CLTV).

Functions:
    home(): Renders the home page of the application.
    predict(): Collects input data from an HTML form, processes it, and uses a pre-trained model to predict the CLTV. 
               The prediction result is then rendered back on the webpage.

Attributes:
    app (Flask): The Flask application instance.
    model: The pre-trained model loaded from a pickle file.

Exceptions:
    If there is an error loading the model or during prediction, an error message is printed or returned as a JSON response.
"""

app = Flask(__name__)

# Load the pickle model
try:
    with open('models/xgbregressor_cltv_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def home():
    """
    Renders the home page of the customer lifecycle management application.
    Returns:
        Response: A Flask response object that renders the "index.html" template.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"]) #Handle POST requests to the /predict endpoint to predict customer lifetime value (CLTV).
def predict():
    """
    This function collects input data from an HTML form, processes it, and uses a pre-trained model
    to predict the CLTV. The prediction result is then rendered back on the webpage.
    Form Data:
        frequency (float): The frequency of purchases.
        total_amount (float): The total amount spent by the customer.
        avg_order_value (float): The average value of an order.
        recency (int): The number of days since the last purchase.
        customer_age (int): The age of the customer.
        lifetime (int): The lifetime of the customer in days.
        purchase_frequency (float): The frequency of purchases over the customer's lifetime.
    Returns:
        Response: A rendered HTML template with the prediction result if successful.
        Response: A JSON object with an error message and a 500 status code if an exception occurs.
    """
    try:
        # Collect input data from the form
        input_data = [
            float(request.form["frequency"]),
            float(request.form["total_amount"]),
            float(request.form["avg_order_value"]),
            int(request.form["recency"]),
            int(request.form["customer_age"]),
            int(request.form["lifetime"]),
            float(request.form["purchase_frequency"]),
        ]
        
        # Make prediction using the loaded model
        predicted_cltv = model.predict([input_data])[0]
        
        # Render the result back on the webpage
        return render_template("index.html", prediction=predicted_cltv)

    except Exception as e:
        # If any error occurs, return the error message
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
