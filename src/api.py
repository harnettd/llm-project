"""Set up a (local) server that takes in POST requests consisting of a 
list of (text) movie reviews and returns a correpsonding classification
of those reviews: a 1 (thumbs-up) for a positive review and a 0 (thumbs-down)
for a negative review. (Note that each movie review's classification is 
actually returned as a pair of probabilities.)   
"""
import torch

from flask import Flask, request
from flask_restful import Resource, Api

from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    TextClassificationPipeline


def main():
    # Load the pretrained tokenizer and model.
    dir = 'movie-review-classifier'
    tokenizer = AutoTokenizer.from_pretrained(f'/home/derek/Documents/Learning/Lighthouse-Labs/Data-Science/projects/llm-project/movie-review-classifier/tokenizer') 
    model = AutoModelForSequenceClassification.from_pretrained(f'/home/derek/Documents/Learning/Lighthouse-Labs/Data-Science/projects/llm-project/movie-review-classifier/model')
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=None  # Return all probabilities.
    )

    app = Flask(__name__)
    api = Api(app)


    class ClassifyMovieReview(Resource):
        """Respond to POST requests of lists of movie reviews."""
        def post(self):
            json_data = request.get_json()
            reviews = json_data['reviews']
            return pipe(reviews) 


    api.add_resource(ClassifyMovieReview, '/classify')
    
    # Start a local server on port 5000.
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
