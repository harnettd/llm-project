"""Set up a (local) server that takes in POST requests consisting of a 
list of (text) movie reviews and returns a correpsonding classification
of those reviews: a 1 (thumbs-up) for a positive review and a 0 (thumbs-down)
for a negative review. (Note that each movie review's classification is 
actually returned as a pair of probabilities.)   
"""
import pickle

from flask import Flask, request
from flask_restful import Resource, Api
from pathlib import Path

from app.cleaner.preprocessor import Preprocessor
from app.cleaner.tokenizer import Tokenizer

app = Flask(__name__)
api = Api(app)

# Load the fitted classifier.
project_path = Path(__file__).parent.parent
model_path = project_path / 'models' / 'best_model.pkl'
with open(model_path, 'rb') as file:
    classifier = pickle.load(file) 

class ClassifyMovieReview(Resource):
    """Respond to POST requests of lists of movie reviews."""
    def post(self):
        json_data = request.get_json()
        reviews = json_data['text']
        return classifier.predict(reviews) 

api.add_resource(ClassifyMovieReview, '/classify')
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
