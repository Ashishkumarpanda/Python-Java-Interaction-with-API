import pickle
from flask import Flask,jsonify,request
import numpy as np


model = pickle.load(open('code\SalaryPredictor.pkl','rb'))


app = Flask(__name__)
@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    year_of_exp = request.form.get('yearOfExperience')
    input_query = np.array([[year_of_exp]]).astype(np.float64)
    salary = model.predict(input_query)[0]

    return jsonify({'Salary':str(np.round(salary))})


   


if __name__ == '__main__':
    app.run(debug=True)