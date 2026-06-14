import os
import traceback

import joblib
import pandas as pd
from fastapi import HTTPException

# Get the directory where this script is located
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Load artifacts
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
model_columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))


def predict_rent(data: dict) -> float:
    try:
        print("Step 1: Input received")

    
        df = pd.DataFrame([data])

        print("Step 2: DataFrame created")

        # Create Period feature
        df["Period"] = df["half_year"].apply(
            lambda x: str(int(x[:4] + x[5]))
        )

        print("Step 3: Period created:", df["Period"].values)

        # Remove original half_year
        df.drop(columns=["half_year"], inplace=True)

        print("Step 4: half_year dropped")

        # One-hot encoding
        categorical_cols = [
            "county",
            "property_type",
            "location",
            "area",
            "province",
            "bedrooms",
            "Period",
        ]

        df_encoded = pd.get_dummies(
            df,
            columns=categorical_cols
        )

        print("Step 5: Dummies created", df_encoded.shape)

        # Convert booleans to integers
        df_encoded = df_encoded.astype(int)

        print("Step 6: Converted to int")

        # Prefer scaler feature names if available
        if hasattr(scaler, "feature_names_in_"):
            expected_columns = list(scaler.feature_names_in_)
            print(
                f"Step 7: Using scaler feature names ({len(expected_columns)} columns)"
            )
        else:
            expected_columns = model_columns
            print(
                f"Step 7: Using columns.pkl ({len(expected_columns)} columns)"
            )

        # Match training columns exactly
        df_encoded = df_encoded.reindex(
            columns=expected_columns,
            fill_value=0
        )

        print("Step 8: Reindex complete", df_encoded.shape)

        # Debug information
        print("Expected columns:", len(expected_columns))
        print("Actual columns:", len(df_encoded.columns))

        # Scale
        scaled = scaler.transform(df_encoded)

        print("Step 9: Scaling complete")

        # Predict
        prediction = model.predict(scaled)

        print("Step 10: Prediction complete:", prediction)

        return float(prediction[0])

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction Error: {str(e)}"
        )