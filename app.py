from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the diabetes dataset
diabetes_df = pd.read_csv('diabetes.csv')

# Extract features and target
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route('/home')
def index():
    return render_template('home.html')

@app.route('/check')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.form.to_dict()
    for key in data:
        data[key] = float(data[key])
    column_names = ['Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for column in column_names:
        if column not in data:
            return "Missing column in input data: {}".format(column)
    input_data = pd.DataFrame([data], columns=column_names)
    imputer = SimpleImputer(strategy='mean')
    input_data_imputed = pd.DataFrame(imputer.fit_transform(input_data), columns=input_data.columns)
    input_data_scaled = scaler.transform(input_data_imputed)
    prediction = model.predict(input_data_scaled)
    return render_template('results.html', prediction='Positive' if prediction[0] == 1 else 'Negative')

if __name__ == '__main__':
    app.run(debug=True)
