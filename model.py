import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings('ignore')


class Visualizer():

    def __init__(self):
        self.df = pd.read_csv(r"merged.csv")

    def plot_scatter(self, x: str, y: str):
        "visualize the correlation between two variables of the dataframe"
        sb.scatterplot(x=x, y=y, data=self.df)

    def plot_distribution(self, x: str):
        "visualize the population distribution of a variable of the dataframe"
        sb.displot(data=self.df, x=x)

class Model:
    def __init__(self, model_type = 'random_forest'):
        "initialize a specific model with model type"

        if model_type == 'random_forest':
            self.model = RandomForestRegressor()
        elif model_type == 'linear_regression':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            raise ValueError("model_type must be 'random_forest' or 'linear_regression'")

        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")  # For prediction time

    def train_model(self, df: pd.DataFrame):
        "train a model on the dataframe"

        X = df.drop(columns=['User_ID', 'Calories'])
        Y = df['Calories']

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

        # Fit imputer on training data to learn median values
        self.imputer.fit(X_train)

        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model.fit(X_train_scaled, y_train)
        train_preds = self.model.predict(X_train_scaled)
        val_preds = self.model.predict(X_val_scaled)

        train_error = mae(y_train, train_preds)
        val_error = mae(y_val, val_preds)

        print(f'Training Error: {train_error:.2f}')
        print(f'Validation Error: {val_error:.2f}')


    def predict_user_calories(self, user_info: dict):
        "Handle missing values in user input during prediction"

        # Check for required fields
        missing_required = []
        required_fields = ['duration', 'workout_type']

        for field in required_fields:
            if field not in user_info or user_info[field] is None:
                missing_required.append(field)

        if missing_required:
            raise ValueError(f"Missing required fields: {missing_required}")


        # Create array with possible None/missing values
        user_data = np.array([[
            user_info.get('age', None),
            user_info.get('gender', None),
            user_info.get('height', None),
            user_info.get('weight', None),
            user_info.get('workout_type'),
            user_info.get('duration'),
            user_info.get('heart_rate', None),
        ]])

        # Convert gender to numeric (handle None)
        if user_data[0, 1] == 'female':
            user_data[0, 1] = 1
        elif user_data[0, 1] == 'male':
            user_data[0, 1] = 0
        else:
            user_data[0, 1] = None  # Will be imputed

        # Impute missing values using the fitted imputer
        user_data_imputed = self.imputer.transform(user_data)

        # Scale the imputed data
        user_data_scaled = self.scaler.transform(user_data_imputed)

        # Make prediction
        predicted_calories = self.model.predict(user_data_scaled)[0]
        return predicted_calories

if __name__ == "__main__":
    df = pd.read_csv(r"merged.csv")
    df.replace({'male': 0, 'female': 1}, inplace=True)

    # Train the model and get the trained model + scaler
    rf = Model(model_type='random_forest')
    rf.train_model(df)

    #rf.predict_user_calories()  # Pass both model and scaler
