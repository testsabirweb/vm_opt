from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

# Load the trained model and features
loaded_classifier = joblib.load('model/decision_tree_model.joblib')
loaded_features = joblib.load('model/model_features.joblib')


def calculate_right_sized(property) -> int:
    property = int(property)
    return property//2 + property//4


def map_value(value, from_min, from_max, to_min, to_max):
    # Ensure the value is within the source range
    value = max(min(value, from_max), from_min)

    # Map the value from the source range to the destination range
    mapped_value = (value - from_min) / (from_max -
                                         from_min) * (to_max - to_min) + to_min

    return round(mapped_value, 3)


def move_to_various_cloud(row):
    cpu_utilization = row[5]
    latency = row[6]
    if row[1] == "On-prem":
        return [
            1.0,
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3),
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3) +
            map_value(latency % 10, 0, 9, -0.3, 0.3),
        ]
    if row[1] == "AWS":
        return [
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3),
            1.0,
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3) +
            map_value(latency % 10, 0, 9, -0.3, 0.3),
        ]
    if row[1] == "GCP":
        return [
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3),
            1.0 + map_value(cpu_utilization % 10, 0, 9, -0.3, 0.3) +
            map_value(latency % 10, 0, 9, -0.3, 0.3),
            1.0,
        ]


def movement_related_calculation(row):
    move_onprem = row[10]
    cpu_utilization = row[5]
    cost = row[7]
    latency = row[6]
    move_aws = row[11]
    move_gcp = row[12]
    updated_latency = int(latency) + \
        map_value(cpu_utilization, 0, 100, -20, 20)
    min_val = min(float(move_onprem), float(move_aws), float(move_gcp))
    if min_val < 0.84:
        if min_val == float(move_onprem):
            return ['On-prem', float(cost) * min_val, int(updated_latency)]
        elif min_val == float(move_aws):
            return ['AWS', float(cost) * min_val, int(updated_latency)]
        else:
            return ['GCP', float(cost) * min_val, int(updated_latency)]

    return ['No movement required', cost, latency]


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
    moved_data = input_data.apply(
        lambda row: move_to_various_cloud(row), axis=1)
    input_data[['move_onprem', 'move_aws', 'move_gcp']] = pd.DataFrame(
        moved_data.tolist(), index=input_data.index)

    moved_data = input_data.apply(
        lambda row: movement_related_calculation(row), axis=1)
    input_data[['where to move', 'Updated Cost', 'Updated latency']
               ] = pd.DataFrame(moved_data.tolist(), index=input_data.index)

    # Save the datasheet with predictions to a new CSV file
    output_csv_path = 'datasets/predicted.csv'
    input_data.to_csv(output_csv_path, index=False)

    return output_csv_path


def get_count_of_cloud(data):
    cloud_count = {}
    cloud_cost = {}

    for index, row in data.iterrows():
        cloud_provider = row['cloud provider']
        total_cost = float(row['Cost Per Year ($)'])

        # Initialize count and cost for the cloud provider if not present
        if cloud_provider not in cloud_count:
            cloud_count[cloud_provider] = 0
            cloud_cost[cloud_provider] = 0.0

        # Increment the count and add to the total cost
        cloud_count[cloud_provider] += 1
        cloud_cost[cloud_provider] += total_cost

    # Round the total cost to 2 digits
    rounded_cloud_cost = {provider: round(
        cost, 2) for provider, cost in cloud_cost.items()}

    # Create a dictionary to hold the result
    result = {'cloud_count': cloud_count,
              'cloud_total_cost': rounded_cloud_cost}
    return result


def top_filter_vm_based_on_cloud(data):
    aws_data = data[data['cloud provider'] == 'AWS'].head(
        5).to_dict(orient='records')
    onprem_data = data[data['cloud provider'] ==
                       'On-prem'].head(5).to_dict(orient='records')
    gcp_data = data[data['cloud provider'] == 'GCP'].head(
        5).to_dict(orient='records')

    result = {
        'AWS': json.loads(json.dumps(aws_data)),
        'On-prem': json.loads(json.dumps(onprem_data)),
        'GCP': json.loads(json.dumps(gcp_data)),
    }

    return result


def filter_vm_based_on_cloud(data):
    aws_data = data[data['cloud provider'] == 'AWS'].to_dict(orient='records')
    onprem_data = data[data['cloud provider'] ==
                       'On-prem'].to_dict(orient='records')
    gcp_data = data[data['cloud provider'] == 'GCP'].to_dict(orient='records')

    result = {
        'AWS': [],
        'On-prem': [],
        'GCP': [],
    }
    for _aws in aws_data:
        if _aws['where to move'] != "No movement required":
            result['AWS'].append(
                f'VM id={_aws["id"]} can be moved to {_aws["where to move"]} ')

    for _on_prem in onprem_data:
        if _on_prem['where to move'] != "No movement required":
            result['On-prem'].append(
                f'VM id={_on_prem["id"]} can be moved to {_on_prem["where to move"]} ')

    for _gcp in gcp_data:
        if _gcp['where to move'] != "No movement required":
            result['GCP'].append(
                f'VM id={_gcp["id"]} can be moved to {_gcp["where to move"]} ')

    return result


def right_sizing_related_data(data):
    aws_data = data[data['cloud provider'] == 'AWS'].to_dict(orient='records')
    onprem_data = data[data['cloud provider'] ==
                       'On-prem'].to_dict(orient='records')
    gcp_data = data[data['cloud provider'] == 'GCP'].to_dict(orient='records')

    result = {
        'AWS': [],
        'On-prem': [],
        'GCP': [],
    }
    for _aws in aws_data:
        if _aws['Predictions'] == "underutilized":
            if int(_aws['Total Ram']) > 1:
                result['AWS'].append(
                    f'VM id={_aws["id"]} reduce Ram to {calculate_right_sized(_aws["Total Ram"])} ')

            if int(_aws['Total CPU']) > 1:
                result['AWS'].append(
                    f'VM id={_aws["id"]} reduce CPU to {calculate_right_sized(_aws["Total CPU"])} ')

        if float(_aws['Cpu Utilization']) > 95:
            result['AWS'].append(
                f'VM id={_aws["id"]} increase CPU to {calculate_right_sized(_aws["Total CPU"])//2 +int(_aws["Total CPU"])} ')

    for _on_prem in onprem_data:
        if _on_prem['Predictions'] == "underutilized":
            if int(_on_prem['Total Ram']) > 1:
                result['On-prem'].append(
                    f'VM id={_on_prem["id"]} reduce Ram to {calculate_right_sized(_on_prem["Total Ram"])} ')

            if int(_on_prem['Total CPU']) > 1:
                result['On-prem'].append(
                    f'VM id={_on_prem["id"]} reduce CPU to {calculate_right_sized(_on_prem["Total CPU"])} ')

        if float(_aws['Cpu Utilization']) > 95:
            result['On-prem'].append(
                f'VM id={_on_prem["id"]} increase CPU to {calculate_right_sized(_on_prem["Total CPU"])//2 +int(_on_prem["Total CPU"])} ')

    for _gcp in gcp_data:
        if _gcp['Predictions'] == "underutilized":
            if int(_gcp['Total Ram']) > 1:
                result['GCP'].append(
                    f'VM id={_gcp["id"]} reduce Ram to {calculate_right_sized(_gcp["Total Ram"])} ')

            if int(_gcp['Total CPU']) > 1:
                result['GCP'].append(
                    f'VM id={_gcp["id"]} reduce CPU to {calculate_right_sized(_gcp["Total CPU"])} ')

        if float(_gcp['Cpu Utilization']) > 95:
            result['GCP'].append(
                f'VM id={_gcp["id"]} increase CPU to {calculate_right_sized(_gcp["Total CPU"])//2 +int(_gcp["Total CPU"])} ')

    return result


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
        # Read the predicted CSV file content
        data = pd.read_csv(predict_using_model(file_path))

        first_page_data = get_count_of_cloud(data)

        # Send both the generated graphs and predictions as a response
        return jsonify({'success': True,
                        'first_page_data': first_page_data
                        })
    else:
        return jsonify({'error': 'Please upload a valid CSV file'})


def get_saved_predictions(file_path, cloud_type):
    data = pd.read_csv(file_path)
    second_page_data = {
        'top_5_filtered_vms_based_on_cloud': top_filter_vm_based_on_cloud(data),
        'moved_vm_data': filter_vm_based_on_cloud(data),
        'right_sized_vm': right_sizing_related_data(data),
    }
    return second_page_data


@app.route('/api/predict/aws', methods=['GET'])
def get_saved_predictions_aws():
    file_path = 'datasets/predicted.csv'
    cloud_type = 'AWS'
    return jsonify({'success': True, 'cloud_type': cloud_type, 'second_page_data': get_saved_predictions(file_path, cloud_type)})


@app.route('/api/predict/on_prem', methods=['GET'])
def get_saved_predictions_on_prem():
    file_path = 'datasets/predicted.csv'
    cloud_type = 'On-prem'
    return jsonify({'success': True, 'cloud_type': cloud_type, 'second_page_data': get_saved_predictions(file_path, cloud_type)})


@app.route('/api/predict/gcp', methods=['GET'])
def get_saved_predictions_gcp():
    file_path = 'datasets/predicted.csv'
    cloud_type = 'GCP'
    return jsonify({'success': True, 'cloud_type': cloud_type, 'second_page_data': get_saved_predictions(file_path, cloud_type)})


if __name__ == '__main__':
    app.run(port=3500, debug=True)
