import CORS
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin

import pandas as pd
import numpy as np
import pickle
import CORS
# Create Flask App
app = Flask(__name__)

# Enable CORS
CORS(app)

# Load Machine Learning Model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load Dataset
car = pd.read_csv('cleaned_Car_data.csv')


# Home Route
@app.route('/', methods=['GET'])
def index():

    companies = sorted(car['company'].unique())

    car_models = sorted(car['name'].unique())

    years = sorted(car['year'].unique(), reverse=True)

    fuel_types = sorted(car['fuel_type'].unique())

    companies.insert(0, 'Select Company')

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )


# Prediction Route
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    company = request.form.get('company')

    car_model = request.form.get('car_models')

    year = request.form.get('year')

    fuel_type = request.form.get('fuel_type')

    driven = request.form.get('kilo_driven')

    # Create DataFrame
    input_data = pd.DataFrame(
        [[car_model, company, year, driven, fuel_type]],
        columns=[
            'name',
            'company',
            'year',
            'kms_driven',
            'fuel_type'
        ]
    )

    # Predict Price
    prediction = model.predict(input_data)

    # Round Prediction
    output = round(prediction[0], 2)

    return str(output)


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)