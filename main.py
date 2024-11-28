from flask import Flask, request, jsonify
from flask_cors import CORS

import xgboost as xgb
import sklearn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import os
import joblib

from data import filter_food, generate_combinations, fetch_nutritions, convert_weight_to_grams, load_and_preprocess_image, safe_convert


app = Flask(__name__)
CORS = CORS(app, origins="*")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
DIABET_FOOD_PATH = os.path.join(DATA_DIR, "diabet_food_recomendation_clean.csv")
NUTRITION_PATH = os.path.join(DATA_DIR, "nutrition.csv")

# Load the saved models and preprocessing tools
exercise_model = joblib.load("./model/exercise/regressor_model.pkl")
classifier = joblib.load("./model/exercise/classifier_model.pkl")
exercise_scaler = joblib.load("./model/exercise/scaler.pkl")
label_encoder = joblib.load("./model/exercise/label_encoder.pkl")

diabetes_model = joblib.load("./model/diabetes/xgb_model_diabetes.pkl")
diabetes_scaler = joblib.load("./model/diabetes/scaler_diabetes.pkl")

food_classification_model = tf.keras.models.load_model("./model/food_classify/FoodImageRecog.h5")

# Get max valuue for diabetes food recommendation
diabet_food_df = pd.read_csv(DIABET_FOOD_PATH)
max_calories = diabet_food_df['Calories'].max()
max_protein = diabet_food_df['Protein'].max()
max_fat = diabet_food_df['Fat'].max()
max_carbs = diabet_food_df['Carbohydrates'].max()

@app.route("/")
def index():
    return "mau ngapain hayo"

# Diabetes prediction
@app.route("/diabetes_predict", methods=["POST"])
def diabetes_predict():
    """
    parameters:
        - features : [gender, age, heart_desease, smoking_history, bmi]
    
    Return: 
        - Prediction Percentage
        - Note
    """

    data = request.json
    gender = data.get('gender')
    age = data.get('age')
    heart_disease = data.get('heart_disease')
    smoking_history = data.get('smoking_history')
    bmi = data.get('bmi')

    if None in [gender, age, heart_disease, smoking_history, bmi]:
        return jsonify({"error": "all fields must be filled"}), 400

    gender = 1 if gender.lower() == 'male' else 0
    smoking_history_mapping = {"never": 0, "current": 1, "former": 2, "ever": 3, not "current": 4}
    smoking_history = smoking_history_mapping.get(smoking_history.lower(), 0)

    features = np.array([[gender, age, heart_disease, smoking_history, bmi]])
    
    try:
        # Rescale the input data
        features_scaled = diabetes_scaler.transform(features)

        prediction_prob = diabetes_model.predict_proba(features_scaled)[0][1]
        prediction_percentage = float(round(prediction_prob * 100, 2))

        response = {
            "percentage": float(f"{prediction_percentage:.2f}"),
            "note": "The patient may be prone to diabetes. Please consult a doctor." if prediction_percentage > 50 else "Patients may not be prone to diabetes."
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Exercise recommendation
@app.route("/exercise_recommendation", methods=["POST"])
def exercise_recomendation():
    """
    Parameters:
        - features: [Gender, Age, Height, Diabetes, BMI]
    
    Returns:
        - Regression predictions (Calories Burned, Exercise Duration)
        - Classification prediction (Predicted Exercise Category)
    """
    data = request.json
    gender = data.get('gender')
    age = data.get('age')
    height = data.get('height')
    diabetes = data.get('diabetes')
    bmi = data.get('bmi')

    # labelencoder
    gender = 1 if gender.lower() == 'male' else 0

    if None  in [gender, age, height, diabetes, bmi]:
        return jsonify({"error": "all fields must be filled"}), 400

    features = [[gender, age, height, diabetes, bmi]]

    # Scale the input data
    scaled_data = exercise_scaler.transform(features)

    try: 
        # Regression Predictions
        regression_results = exercise_model.predict(scaled_data)
        
        # Classification Predictions
        classification_results = classifier.predict(scaled_data)
        decoded_class = label_encoder.inverse_transform(classification_results)
        exercise_categories = list(decoded_class[0].split(","))

        exercises = []
        for category in exercise_categories:
            exercises.append(category.replace("or", " ").strip().capitalize().replace(".", ""))

        response = {
            "calories_burned": regression_results[0][0],  
            "exercise_categories": exercises,
            "exercise_duration": round(regression_results[0][1], 2)
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Food recommendation
@app.route("/food_recommendation", methods=["POST"])
def food_recommendation():
    """
    Parameters:
        - diabetes_percentage : 0-100 (percentage)

    Returns:
        - Food recommendation based on diabetes or not diabetes
    """
    data = request.json
    diagnoses = data.get("diabetes_percentage")

    if diagnoses is None:
        return jsonify({"error": "The 'diabetes' field must be provided"}), 400

    try:
        diagnoses = float(diagnoses) / 100
        diagnoses = 1 if diagnoses >= 0.5 else 0

        print(diagnoses)
    except (ValueError, TypeError):
        return jsonify({"error": "The 'diabetes' field must be a numeric value"}), 400

    try:
        if not os.path.exists(DIABET_FOOD_PATH) or not os.path.exists(NUTRITION_PATH):
            return jsonify({"error": "Required data files not found"}), 500

        food_df = pd.read_csv(NUTRITION_PATH)

        diabetes_food = filter_food(
            food_df, max_calories=max_calories,
            max_carbohydrate=max_carbs, max_fat=max_fat, max_protein=max_protein
        )
        normal_food = food_df

        diabetes_combinations = generate_combinations(diabetes_food)
        normal_combinations = generate_combinations(normal_food)

        food_recommendation = [
            combo[["name", "calories", "carbohydrate", "fat", "proteins", "image"]].to_dict(orient="records")
            for combo in (diabetes_combinations if diagnoses else normal_combinations)
        ]

        response = {
            "diabetes": bool(diagnoses),
            "food_recommendation": food_recommendation
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Food clasifications
@app.route("/food_classification", methods=["POST"])
def food_clasification():
    """
    Parameters:
        - features : [images of food, name food, volume]

    Returns:
        - food name
        - [Calories, Carbohydrates, Fat, Proteins]
        - alert (note recommendation for diabetes or not diabetes)
        - volume (gram)
    """
    
    data = request.json
    image = data.get('image')
    food_name = data.get('name')
    volume = data.get('volume')

    class_names = ['Abon', 'Chiken', 'Cumi', 'Beef', 'Gorengan', 'Ikan', 'Kerupuk', 'Mie',
                   'Rice', 'shrimp', 'Sambal', 'Sate', 'Sayur Hijau', 'Tempeh', 'Egg',
                   'Tofu', 'Olahan Daging']

    if not image and not food_name:
        return jsonify({"error": "Either 'image' or 'name' must be provided."}), 400

    if image is not None:
        img_array = load_and_preprocess_image(image)

        # Make a prediction using the model
        predictions = food_classification_model.predict(img_array)

        # Get the predicted class (the index with the highest probability)
        predicted_class_idx = np.argmax(predictions, axis=-1)

        # Print the prediction result
        food_name = class_names[predicted_class_idx[0]]

    if not food_name:
        return jsonify({"error": "Food name could not be determined."}), 400
    
    if volume is not None:
        volume_convert = convert_weight_to_grams(volume)

    try:
        proteins, calories, carbohydrates, fat = fetch_nutritions(food_name)

        # Convert values to floats to avoid type mismatch
        proteins = safe_convert(proteins, "g")
        calories = safe_convert(calories, "kcal")
        carbohydrates = safe_convert(carbohydrates, "g")
        fat = safe_convert(fat, "g")
        
        # Ensure volume is a valid number
        volume_convert = float(volume_convert / 100) if volume is not None else 1

        proteins *= volume_convert
        calories *= volume_convert
        carbohydrates *= volume_convert
        fat *= volume_convert

        nutrition_info = {
            "proteins": "{:.2f} g".format(proteins),
            "calories": "{:.2f} kcal".format(calories),
            "carbohydrates": "{:.2f} g".format(carbohydrates),
            "fat": "{:.2f} g".format(fat)
        }

        alert = (
            "Suitable for diabetes" 
            if (carbohydrates < max_carbs and 
                calories < max_calories and 
                proteins < max_protein and 
                fat < max_fat)
            else "Not recommended for diabetes"
        )
        
        return jsonify({
            "food_name": food_name,
            "nutrition_info": nutrition_info,
            "alert": alert,
            "volume": "100 g" if volume is None else f"{volume}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
