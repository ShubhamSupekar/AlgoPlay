from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pandas as pd 
import model.LinearRegression.LinearRegression as lr 

# Create the FastAPI app instance
app = FastAPI()

# Store the dataframe globally after loading
df = None


datasets = {
    "canada-house-price": "datasets\Linear Regression\housing.csv",
    "laptop-price": "datasets\Linear Regression\laptop_price - dataset.csv",
    "customer-behaviour": "datasets\Linear Regression\Customer Purchasing Behaviors.csv"
}

# "static" directory to serve CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates object
templates = Jinja2Templates(directory="templates")



# home page
@app.get("/")
async def home(request: Request):
    # Render the home page with no message initially

    return templates.TemplateResponse("index.html", {
        "request": request,
        "selected_dataset": None  # No dataset selected initially
    })


# @app.post("/train")
# async def train_model(dataset: str = Form(...)):
#     global df
#     if df is not None:
#         try:
#             lr.LoadDataset(df)
#             message = f"Model training started for dataset '{dataset}'."
#         except Exception as e:
#             message = f"Error during training: {str(e)}"
#     else:
#         message = "No dataset loaded for training."

#     # Redirect back to the homepage with a message
#     return RedirectResponse(url=f"/?message={message}", status_code=303)



@app.post("/process-dataset")
async def process_dataset(request: Request, dataset: str = Form(...)):
    file_path = datasets.get(dataset)
    
    if file_path:
        try:
            # Load the selected dataset
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()  # Convert Index object to a list
            message = f"Dataset '{dataset}' successfully loaded."
        except Exception as e:
            message = f"Error loading dataset: {str(e)}"
            columns = []
    else:
        message = "Invalid dataset selected."
        columns = []

    # Render the response with the success or failure message and keep the selected dataset
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "message": message,
        "selected_dataset": dataset,  # Pass the selected dataset back to the template
        "columns": columns  # Pass columns as a list to the template
    })

