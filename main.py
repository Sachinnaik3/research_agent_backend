from app.api.routers import research_agent
from fastapi import FastAPI
from app.utils.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="Research Agent API", description="API for Research Agent using LLM", version="1.0")
app.mount("/images", StaticFiles(directory="images"), name="images")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(research_agent.router)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Research Agent API!"}


