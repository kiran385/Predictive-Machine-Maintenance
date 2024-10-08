from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and prepare the dataset
data = pd.read_csv('predictive_maintenance.csv')

# Assuming 'Target' is the column with labels and the rest are features
X = data.drop('Target', axis=1)
y = data['Target']

# Convert categorical features to numeric using One-Hot Encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest model and train it
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form values from the user input
    input_data = request.form.to_dict()
    
    # Create a DataFrame with the same structure as training data
    features = {
        'Type': input_data['Type'],  # Keep 'Type' as a string
        'Air temperature [K]': float(input_data['Air temperature [K]']),
        'Process temperature [K]': float(input_data['Process temperature [K]']),
        'Rotational speed [rpm]': float(input_data['Rotational speed [rpm]']),
        'Torque [Nm]': float(input_data['Torque [Nm]']),
        'Tool wear [min]': float(input_data['Tool wear [min]'])
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([features])
    
    # Apply one-hot encoding to match the training set's feature structure
    input_df = pd.get_dummies(input_df)
    
    # Reindex to ensure all features are present
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Use the model to predict the result
    prediction = model.predict(input_df)

    # Convert prediction to human-readable format
    if prediction[0] == 1:
        result_text = 'Maintenance is needed'
    else:
        result_text = 'No maintenance needed'

    # Return the prediction to the frontend
    return render_template('index.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True)
