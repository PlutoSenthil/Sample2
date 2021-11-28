from fastapi import FastAPI
from model_test import Test
app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

app.include_router(Test.router)

@app.get('/')
async def home():
    return {'Welcome':'Home Page'}