from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Create the FastAPI app instance
app = FastAPI()

# Mount the "static" directory to serve CSS and other static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create the Jinja2 templates object
templates = Jinja2Templates(directory="templates")

# Define a route for the home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
