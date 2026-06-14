from fastapi import FastAPI
from schema.user_input import UserInput
from model.predict import predict_rent

app = FastAPI()




@app.get("/")
def home():
    return {"Welcome to Irish rent prediction API"}

@app.get("/health")
def health():
    return {"status": "ok",
           "message": "API is healthy and running"}

@app.post("/predict")
def predict(input_data: UserInput):
    import traceback
    try:
        result = predict_rent(input_data.dict())
        return {"predicted_rent_euro": result}
    except Exception as e:
        traceback.print_exc()
        raise
