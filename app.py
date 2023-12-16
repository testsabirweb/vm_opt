from flask import Flask, request, jsonify
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Load the trained model and features
loaded_classifier = joblib.load('model/decision_tree_model.joblib')
loaded_features = joblib.load('model/model_features.joblib')

def predict_using_model(input_csv_path):
    # Load the input CSV file
    input_data = pd.read_csv(input_csv_path)

    # Select relevant features from the input datasheet
    X_input = input_data[loaded_features]

    # Make predictions using the loaded model
    predictions = loaded_classifier.predict(X_input)

    # Add predictions to the input datasheet
    input_data['Predictions'] = predictions

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
