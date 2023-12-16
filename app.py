from flask import Flask, request, jsonify
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Load the trained model and features
loaded_classifier = joblib.load('model/decision_tree_model.joblib')
loaded_features = joblib.load('model/model_features.joblib')

def map_value(value, from_min, from_max, to_min, to_max):
    # Ensure the value is within the source range
    value = max(min(value, from_max), from_min)
    
    # Map the value from the source range to the destination range
    mapped_value = (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
    
    return round(mapped_value,3)
    
def move_to_various_cloud(row):
    cpu_utilization=row[5]
    if row[1]=="On-prem":
        return[1.0,1.0+map_value(cpu_utilization,0,100,0.1,0.15),1.0+map_value(cpu_utilization,0,100,0.15,0.25)]
    if row[1]=="AWS":
        return[1.0-map_value(cpu_utilization,0,100,0.1,0.15),1.0,1.0+map_value(cpu_utilization,0,100,0.1,0.15)]
    if row[1]=="GCP":
        return[1.0-map_value(cpu_utilization,0,100,0.15,0.25),1.0-map_value(cpu_utilization,0,100,0.05,0.15),1.0]
    
def predict_using_model(input_csv_path):
    # Load the input CSV file
    input_data = pd.read_csv(input_csv_path)

    # Select relevant features from the input datasheet
    X_input = input_data[loaded_features]

    # Make predictions using the loaded model
    predictions = loaded_classifier.predict(X_input)

    # Add predictions to the input datasheet
    input_data['Predictions'] = predictions
    # create columns for movement logic
    moved_data= input_data.apply(lambda row: move_to_various_cloud(row), axis=1)
    input_data[['move_onprem', 'move_aws', 'move_gcp']] = pd.DataFrame(moved_data.tolist(), index=input_data.index)

    # Save the datasheet with predictions to a new CSV file
    output_csv_path = 'datasets/predicted.csv'
    input_data.to_csv(output_csv_path, index=False)

    return output_csv_path

@app.route('/api/predict', methods=['POST'])
def predict():
    # Check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    # Check if the file is of CSV format
    if file and file.filename.endswith('.csv'):
        # Save the uploaded CSV file
        file_path = 'uploads/' + file.filename
        file.save(file_path)

        # Perform analysis using the trained model
        output_csv_path = predict_using_model(file_path)

        # Read the predicted CSV file content
        predicted_data = pd.read_csv(output_csv_path)

        # Convert the DataFrame to JSON for the API response
        predicted_json = predicted_data.to_json(orient='records')

        return jsonify({'success': True, 'data': predicted_json})

    else:
        return jsonify({'error': 'Please upload a valid CSV file'})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
