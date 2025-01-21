from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

class CodeMatcher:
    def __init__(self, codes):
  
        self.df = pd.read_csv(codes)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['description'])
    
    def match_code(self, input_description):

        input_vec = self.vectorizer.transform([input_description])
        similarity = cosine_similarity(input_vec, self.tfidf_matrix)
        best_match_idx = similarity.argmax()
        return self.df.iloc[best_match_idx]['code']


codes_csv_path = 'codes.csv'

if not os.path.exists(codes_csv_path):
    raise FileNotFoundError(f"The CSV file at {codes_csv_path} does not exist.")
matcher = CodeMatcher(codes_csv_path)

app.route('/')
def home():
    return "ONLINE"

@app.route('/match', methods=['POST'])
def match():
    """
    API endpoint to match descriptions to codes.
    Expects a JSON body with a list of descriptions.
    PAYLOAD EXAMPLE : 
            {
        "descriptions": [
            "Handling",
            "Weigh Bridge"]
}
    """
    content = request.json
    descriptions = content.get('descriptions', [])
    
    if not descriptions or not isinstance(descriptions, list):
        return jsonify({"error": "Invalid input. Provide a list of descriptions."}), 400

    results = {}
    for description in descriptions:
        matched_code = matcher.match_code(description)
        results[description] = matched_code
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True,port=3000)
