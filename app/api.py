"""Set up a (local) server that takes in POST requests consisting of a 
list of (text) movie reviews and returns a corresponding classification
of those reviews: a 1 (thumbs-up) for a positive review and a 0 (thumbs-down)
for a negative review.   
"""
import pickle

from flask import Flask
from flask_restful import Resource, Api, reqparse
from pathlib import Path

from app.cleaner.preprocessor import Preprocessor
from app.cleaner.tokenizer import Tokenizer

app = Flask(__name__)
api = Api(app)

# Load the fitted classifier.
project_path = Path(__file__).parent.parent
model_path = project_path / 'app' / 'model' / 'best_model.pkl'
with open(model_path, 'rb') as file:
    classifier = pickle.load(file) 

parser = reqparse.RequestParser()
parser.add_argument('reviews', type=list, location='json', required=True,
                    help='List of movie reviews cannot be parsed')

class ClassifyMovieReviews(Resource):
    """Respond to POST requests of lists of movie reviews."""
    def post(self):
        args = parser.parse_args()
        reviews = args['reviews']
        return {'labels': classifier.predict(reviews).tolist()}, 200

api.add_resource(ClassifyMovieReviews, '/classify')
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
