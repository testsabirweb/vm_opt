from flask import Flask, request, jsonify
import pandas as pd
import joblib
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the trained model and features
loaded_classifier = joblib.load('model/decision_tree_model.joblib')
loaded_features = joblib.load('model/model_features.joblib')


def plot_costs_side_by_side(input_data, output_folder='graphs'):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the total cost and total updated cost
    total_cost = input_data['Cost per Day ($)'].sum()
    total_updated_cost = input_data['Updated Cost'].sum()

    # Calculate the difference in cost
    cost_difference = total_updated_cost - total_cost

    # Create a bar graph for Total Cost and Updated Cost side by side
    fig, ax = plt.subplots()
    bar_width = 0.35

    # Set positions for the bars
    positions = [0, 1]  # Numeric positions for Total Cost and Total Updated Cost
    bar1 = ax.bar(positions[0], total_cost, bar_width, label='Total Cost')
    bar2 = ax.bar(positions[1], total_updated_cost, bar_width, label='Total Updated Cost')

    # Set labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel('Cost ($)')
    ax.set_title('Total Cost vs Total Updated Cost')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Total Cost', 'Total Updated Cost'])
    ax.legend()

    # Add a text annotation for the difference in cost
    ax.text(positions[1], total_updated_cost + 1, f'Difference: {cost_difference:.2f}', ha='center', va='bottom', color='red')

    # Save the plot to the output folder
    output_path = os.path.join(output_folder, 'total_costs_side_by_side.png')
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying it if not needed


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
    
def movement_related_calculation(row):
    move_onprem=row[10]
    cpu_utilization=row[5]
    cost=row[7]
    latency=row[6]
    move_aws=row[11]
    move_gcp=row[12]
    if  row[1]!="On-prem":
        if float(move_onprem)<0.78:
            updated_latency=int(latency)+map_value(cpu_utilization,0,100,10,20)
            return ["On-Prem",float(cost)*float(move_onprem),int(updated_latency)]
    return ["No movement required",cost,latency]
    
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

    moved_data= input_data.apply(lambda row: movement_related_calculation(row), axis=1)
    input_data[['where to move', 'Updated Cost','Updated latency']] = pd.DataFrame(moved_data.tolist(), index=input_data.index)
    # Save the datasheet with predictions to a new CSV file
    output_csv_path = 'datasets/predicted.csv'
    
    # Plot costs and save the graph
    plot_costs_side_by_side(input_data)
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
    app.run(port=3500, debug=False)
