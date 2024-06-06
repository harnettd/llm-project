import torch

from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    TextClassificationPipeline


def main():
    # Load the pretrained tokenizer and model.
    dir = 'movie-review-classifier'
    tokenizer = AutoTokenizer.from_pretrained(f'{dir}/tokenizer') 
    model = AutoModelForSequenceClassification.from_pretrained(f'{dir}/model')

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=None  # Return all probabilities.
    )

    app = Flask(__name__)
    api = Api(app)


    class ClassifyMovieReview(Resource):
        def post(self):
            json_data = request.get_json()
            reviews = json_data['reviews']
            return pipe(reviews) 


    api.add_resource(ClassifyMovieReview, '/classify')
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
