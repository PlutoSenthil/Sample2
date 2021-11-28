import re
from string import punctuation

import joblib
from fastapi import APIRouter
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

router =APIRouter(
    prefix="/model",
    tags=['Output']
)


import configparser
config_object= configparser.ConfigParser()
config_object.read("configVariable.ini")
with open(config_object['Path']['output_model_path'], "rb") as f:
    model = joblib.load(f)

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text


@router.get("/predict-review")
async def predict_sentiment(review: str):
    # clean the review
    cleaned_review = text_cleaning(review)
     # perform prediction
    prediction = model.predict([cleaned_review])
    probas = model.predict_proba([cleaned_review])
    output = int(prediction[0])
    print(output)
    output_probability = "{:.2f}".format(float(probas[:, output]))
    sentiments = {0: "Negative", 1: "Positive"}
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result