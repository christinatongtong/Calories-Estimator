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
        user_data = {}
        user_data['age'] = request.form['age']
        user_data['gender'] = request.form['gender']
        user_data['height'] = request.form['height']
        user_data['weight'] = request.form['weight']
        user_data['workout_type'] = request.form['workout_type']
        user_data['duration'] = request.form['duration']

        if user_data['duration'] == 0:
            return render_template('result.html', predicted_calories=0)
        user_data['heart_rate'] = request.form['heart_rate']

        predicted_calories = model.predict_user_calories(user_data)
        return render_template('result.html', prediction=predicted_calories, user_info=user_data)

    except Exception as e:
        return render_template('result.html', prediction=e, user_info=user_data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
