# Irish Rent Prediction API

A machine learning API that predicts rental prices in Ireland based on 
location and property details. Built with FastAPI and deployed using Docker.

## About the Project

Rent prices in Ireland vary massively depending on location, property type 
and time period. I built a Linear Regression model that captures these 
patterns across different counties and provinces, with special attention 
to whether a property is in Dublin or not — since Dublin skews the entire 
rental market.

**Model R² Score: 0.91**

## Tech Stack

- Python, Scikit-learn, Pandas
- FastAPI
- Docker & Docker Hub

## Input Features

| Feature | Description |
|---------|-------------|
| year | Year of the rental listing |
| half | First or second half of the year |
| half_year | Combined half-year period |
| time_period | Overall time period |
| county | County in Ireland |
| province | Province (Leinster, Munster, etc.) |
| area | Area within the county |
| location | Specific location |
| property_type | Type of property (apartment, house, etc.) |
| bedrooms | Bedroom category |
| bedrooms_num | Number of bedrooms |
| is_dublin | Whether property is in Dublin (0/1) |
| is_city | Whether location is a city (0/1) |
| is_county_aggregate | Whether data is county level (0/1) |

## How to Run Locally

**Clone the repo**
```bash
git clone https://github.com/Tanishq1-bit/irish-rent-prediction-api.git
cd irish-rent-prediction-api
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the API**
```bash
uvicorn main:app --reload
```

API will be live at `http://localhost:8000`

## API Usage

Send a POST request to `/predict` with the following JSON:

```json
{
  "year": 2023,
  "half": 1,
  "half_year": "2023H1",
  "time_period": "Q1",
  "county": "Dublin",
  "province": "Leinster",
  "area": "Dublin City",
  "location": "Dublin 2",
  "property_type": "apartment",
  "bedrooms": "2 bed",
  "bedrooms_num": 2,
  "is_dublin": 1,
  "is_city": 1,
  "is_county_aggregate": 0
}
```

Response:
```json
{
  "predicted_rent": 2350.75
}
```

## API Docs

Once running, visit `http://localhost:8000/docs` for the interactive 
Swagger UI to test all endpoints.
