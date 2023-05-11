from flask import Flask
import redis
import pickle

import logging
import time


app = Flask(__name__)
cache = redis.Redis(host="localhost", port=6379)


def load_model():
    from transformers import pipeline

    model_name = "twmkn9/albert-base-v2-squad2"
    device = "cpu"
    logging.info(f"Loading the {model_name} model")
    logging.info(f"Device: {device}")
    start = time.time()
    model = pipeline("question-answering", model=model_name, device=device)
    logging.info(f"Model {model_name} loaded in:{(time.time())-start} seconds")
    return model


@app.before_request
def cache_models():
    if not cache.exists("model"):
        model = load_model()
        pickled_model = pickle.dumps(model)
        cache.set("model", pickled_model)
        cache.set(
            "model", pickled_model, ex=3600
        )  # Cache expires in 1 hour (3600 seconds)


@app.route("/answer")
def predict():
    # Retrieve the model from the cache
    pickled_model = cache.get("model")
    model = pickle.loads(pickled_model)

    return
    # Use the model for prediction
    # ...


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
