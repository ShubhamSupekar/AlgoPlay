from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd 
import model.LinearRegression.LinearRegression as lr 

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

@app.post("/train")
async def train_model(request: Request, predictor_column: str = Form(...)):
    global df  # Access the global df variable
    # Validate the column selected
    if df is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "No dataset loaded. Please load a dataset first."
        })

    if predictor_column not in df.columns:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Invalid predictor column selected."
        })
    
    best_features, best_r2 = lr.StartTraining(predictor_column, df)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "best_features": best_features,
        "best_r2": best_r2
    })
