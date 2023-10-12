from flask import Flask, request, jsonify
import numpy as np
import pickle

# Starting FastApi
app = Flask(__name__)

# Loading k-means model
with open("nasa2.pkl", "rb") as model_file:
    model = pickle.load(model_file)

class InputData:
    def __init__(self, UTC, Temperature_C, Humidity_percent, TVOC_ppb, eCO2_ppm, Raw_H2, Raw_Ethanol, Pressure_hPa, PM1_0, PM2_5, NC0_5, NC1_0, NC2_5, CNT):
        self.UTC = float(UTC)
        self.Temperature_C = float(Temperature_C)
        self.Humidity_percent = float(Humidity_percent)
        self.TVOC_ppb = float(TVOC_ppb)
        self.eCO2_ppm = float(eCO2_ppm)
        self.Raw_H2 = float(Raw_H2)
        self.Raw_Ethanol = float(Raw_Ethanol)
        self.Pressure_hPa = float(Pressure_hPa)
        self.PM1_0 = float(PM1_0)
        self.PM2_5 = float(PM2_5)
        self.NC0_5 = float(NC0_5)
        self.NC1_0 = float(NC1_0)
        self.NC2_5 = float(NC2_5)
        self.CNT = float(CNT)


@app.route('/')
def index():
    return 'Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get json data
        data = request.get_json()

        # Create input data object
        input_data = InputData(
            data['UTC'],
            data['Temperature_C'],
            data['Humidity_percent'],
            data['TVOC_ppb'],
            data['eCO2_ppm'],
            data['Raw H2'],
            data['Raw Ethanol'],
            data['Pressure_hPa'],
            data['PM1.0'],
            data['PM2.5'],
            data['NC0.5'],
            data['NC1.0'],
            data['NC2.5'],
            data['CNT']
        )

        # Use input data for clustering
        input_features = [
            input_data.UTC,
            input_data.Temperature_C,
            input_data.Humidity_percent,
            input_data.TVOC_ppb,
            input_data.eCO2_ppm,
            input_data.Raw_H2,
            input_data.Raw_Ethanol,
            input_data.Pressure_hPa,
            input_data.PM1_0,
            input_data.PM2_5,
            input_data.NC0_5,
            input_data.NC1_0,
            input_data.NC2_5,
            input_data.CNT
        ]

        predicted = int(model.predict([input_features])[0])
        probability = model.predict_proba([input_features])[0][1] * 100
        return jsonify({"predicted": predicted, "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.1',port=5000)
