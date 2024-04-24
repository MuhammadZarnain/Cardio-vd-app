from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the prediction model
model = pickle.load(open('./heart_predict.pkl', 'rb'))

@app.route('/predictheart', methods=['POST'])
def predictheart():
    Age= int(request.form['Age'])
    Sex= int(request.form['Sex'])
    ChestPainType= int(request.form['ChestPainType'])
    RestingBP= int(request.form['RestingBP'])
    Cholesterol= int(request.form['Cholesterol'])
    RestingECG= int(request.form['RestingECG'])
    ExerciseAngina= int(request.form['ExerciseAngina'])

    input_data = ( Age,Sex, ChestPainType, RestingBP,Cholesterol,RestingECG,ExerciseAngina)
    input_data_array= np.asarray(input_data)
    input_data_reshape=input_data_array.reshape(1,-1)
    prediction= model.predict(input_data_reshape)[0]
    return jsonify({'Heart Disease':str(prediction)})
    
    

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)