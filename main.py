from typing import List
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
import pandas as pd 
import model.Regression as lr 
import numpy as np
from sklearn.linear_model import LinearRegression

# Create the FastAPI app instance
app = FastAPI()

# Global variable to store the DataFrame
df = None

# Datasets mapping
datasets = {
    "canada-house-price": "datasets/Linear Regression/housing.csv",
    "laptop-price": "datasets/Linear Regression/laptop_price - dataset.csv",
    "customer-behaviour": "datasets/Linear Regression/Customer Purchasing Behaviors.csv"
}

# Static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates object
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "selected_dataset": None
    })



# Data processing
@app.post("/process-dataset")
async def process_dataset(request: Request, dataset: str = Form(...)):
    global df  # Use the global df variable
    file_path = datasets.get(dataset)

    if file_path:
        try:
            # Load the selected dataset
            df = pd.read_csv(file_path)
            columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            df = df[columns]  # Filter to include only numeric columns
            message = f"Dataset '{dataset}' successfully loaded."
        except Exception as e:
            message = f"Error loading dataset: {str(e)}"
            columns = []
    else:
        message = "Invalid dataset selected."
        columns = []

    # Render the response with the message and selected dataset
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": message,
        "selected_dataset": dataset,
        "columns": columns
    })



# Removing unwanted columns from dataset 
@app.post("/remove-columns")
async def remove_columns(request: Request, columns_to_remove: List[str] = Form(...), dataset: str = Form(...)):
    global df
    if df is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No dataset loaded. Please load a dataset first."
        })
    
    for clm in columns_to_remove:
        if clm not in df.columns:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Invalid predictor column selected."
            })
    
    df = df.drop(columns=columns_to_remove, axis=1)  # Corrected parameter
    message = f"{columns_to_remove} columns have been dropped from the dataset."
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "RemoveClmMsg": message,  # Added missing comma here
        "dataset_used": dataset,  # Show which dataset was used in training
        "selected_dataset": dataset,
        "columns": df.columns.tolist()  # Corrected method to get the list of columns
    })


@app.post("/train")
async def train_model(request: Request, predictor_column: str = Form(...), dataset: str = Form(...)):
    global df  # Access the global df variable

    # Validate if a dataset is loaded
    if df is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No dataset loaded. Please load a dataset first."
        })

    # Validate the selected predictor column
    if predictor_column not in df.columns:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Invalid predictor column selected."
        })

    # Train the model using the selected predictor column
    results = lr.StartTraining(predictor_column, df)

    # Prepare the output for each model in a list format
    model_results = [
        {
            "model": result[0],
            "features": result[1],
            "r2_score": result[2]
        } for result in results
    ]

    # Sort model results by R² score in decreasing order
    model_results.sort(key=lambda x: x['r2_score'], reverse=True)

    # Find the best model based on R² score (first in sorted list)
    best_model_result = model_results[0] if model_results else None
    
    # Prepare a list of models excluding the best model for display
    other_models = model_results[1:] if model_results else []

    # Return the template with the best model and all models
    return templates.TemplateResponse("index.html", {
        "request": request,
        "predict_column": predictor_column,
        "best_model": best_model_result['model'] if best_model_result else None,  # Best model name
        "best_features": best_model_result['features'] if best_model_result else None,  # Best features
        "best_r2": best_model_result['r2_score'] if best_model_result else None,  # Best R² score
        "other_models": other_models,  # All other models
        "dataset_used": dataset,  # Show which dataset was used in training
        "selected_dataset": dataset
    })







# Model to return results
class LinearRegressionResult(BaseModel):
    slope: float
    intercept: float


# Serve the Learn page
@app.get("/learn")
async def get_learn_page(request: Request):
    return templates.TemplateResponse("learn.html", {"request": request})


# Endpoint to run the Python code and return the result
@app.post("/run-python", response_model=LinearRegressionResult)
async def run_python():
    # Data
    x = np.array([2, 4, 6, 8]).reshape(-1, 1)
    y = np.array([3, 7, 5, 10])

    # Create and fit the model
    model = LinearRegression()
    model.fit(x, y)

    # Get the slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # Return the results
    return LinearRegressionResult(slope=slope, intercept=intercept)