from flask import Flask, render_template, request, jsonify
import torch
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor
import work.dataset_reader  # Ensure this module is correctly implemented
import work.text_classifier  # Ensure this module is correctly implemented
import work.models  # Ensure this module is correctly implemented
from work.text_classifier import ComplexTextClassifier
import matplotlib.pyplot as plt
import os
import pandas as pd
archive = load_archive(
        "C:/Users/reddyk6780/Downloads/model (3).tar.gz",
        overrides='{"dataset_reader.type": "my_dataset_reader"}'
    )
predictor = Predictor.from_archive(archive, predictor_name="text_classifier")

device = torch.device("cpu")
predictor._model.to(device)
app = Flask(__name__)

index_to_rating = {
        3: 3,
        4: 0,
        2: 2,
        0: 4,
        1: 1
    }

@app.route('/')
def login():
    return render_template('login_page.html')

@app.route('/home_page')
def home():
    return render_template('home_page.html')

@app.route('/review')
def review():
    return render_template('index.html') 

@app.route('/logout')
def logout():
    return render_template('login_page.html')
@app.route('/dashboard')
def dashboard():
    return render_template('upload.html')
@app.route('/predict', methods=['POST'])
def predict():
    text = {"sentence":request.form['sentence']}
    prediction = predictor.predict_json(text)

    max_index = prediction['probs'].index(max(prediction['probs']))
    print(prediction['probs'])
    rating = index_to_rating[max_index]

    print("Predicted Rating:", rating)
    return jsonify({"rating": rating}) 

    # return render_template('index.html', message=rating)
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save and read CSV file
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)

    if 'comment_text' not in df.columns:
        return "CSV must contain a 'comment_text' column", 400

    # Predict ratings for all reviews
    def get_rating(text):
        prediction = predictor.predict_json({"sentence": text})
        probs = prediction.get('probs', [])
        
        if not probs:
            return None  # Handle cases where 'probs' is missing or empty
        
        max_index = probs.index(max(probs))  # Get the index of the max probability
        
        return index_to_rating.get(max_index, None)  # Use .get() to avoid KeyErrors

    df['rating'] = df['comment_text'].apply(get_rating)

    # Generate rating distribution plot
    rating_counts = df['rating'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(rating_counts.index, rating_counts.values, color='skyblue')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Rating Distribution')
    plt.xticks(range(min(index_to_rating.values()), max(index_to_rating.values()) + 1))
    
    plot_path = os.path.join("static", "rating_distribution.png")
    plt.savefig(plot_path)
    plt.close()

    return render_template('results.html', image_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
