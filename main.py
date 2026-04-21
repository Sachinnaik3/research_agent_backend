from app.api.routers import research_agent
from fastapi import FastAPI
from app.utils.logger import logger

app = FastAPI(title="Research Agent API", description="API for Research Agent using LLM", version="1.0")

# Include API routers
app.include_router(research_agent.router)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the Research Agent API!"}


