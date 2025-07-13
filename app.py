from flask import Flask, render_template, request, jsonify
from model import Model
import pandas as pd


app = Flask(__name__)

def load_model():
    model = Model(model_type='random_forest')
    df = pd.read_csv(r"merged.csv")
    df.replace({'male': 0, 'female': 1}, inplace=True)
    model.train_model(df)
    return model

model = load_model()

# when the original url is visited, the index.html file is rendered
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        user_info = {}
        user_info['age'] = request.form['age']
        user_info['gender'] = request.form['gender']
        user_info['height'] = request.form['height']
        user_info['weight'] = request.form['weight']
        user_info['workout_type'] = request.form['workout_type']
        user_info['duration'] = request.form['duration']
        if user_info['duration'] == 0:
            return render_template('result.html', predicted_calories=0)
        user_info['heart_rate'] = request.form['heart_rate']

        predicted_calories = model.predict_user_calories(user_info)
        return render_template('result.html', predicted_calories=predicted_calories)

    except Exception as e:
        return render_template('result.html', predicted_calories=e)
