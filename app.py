from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

app = Flask(__name__)

# Load and prepare the dataset
data = pd.read_csv('predictive_maintenance.csv')

# Separate features and target variable
X = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = data['Failure Type']

# Encode categorical features
label_encoder = LabelEncoder()
X.loc[:, 'Type'] = label_encoder.fit_transform(X['Type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder
joblib.dump(model, 'dtc.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = request.form.to_dict()

    # Get values from the form
    input_type = input_data['Type']
    air_temp = float(input_data['Air temperature [K]'])
    process_temp = float(input_data['Process temperature [K]'])
    rotational_speed = float(input_data['Rotational speed [rpm]'])
    torque = float(input_data['Torque [Nm]'])
    tool_wear = float(input_data['Tool wear [min]'])

    # Load the trained model and label encoder
    model = joblib.load('dtc.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Encode the 'Type' input
    input_type_encoded = label_encoder.transform([input_type])[0]

    # Create a DataFrame for the input features
    input_features = {
        'Type': input_type_encoded,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    input_df = pd.DataFrame([input_features])

    # Make the prediction
    prediction = model.predict(input_df)[0]

    # Convert prediction to readable format
    result_text = f'Predicted Failure Type: {prediction}'

    # Render the template with both the prediction and the input values
    return render_template('index.html',prediction_text=result_text,input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
