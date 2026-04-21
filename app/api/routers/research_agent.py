from fastapi import HTTPException,APIRouter
from pydantic import BaseModel
from app.agents.research_agent import generate_response
from app.utils.logger import logger
import asyncio

router = APIRouter()

# request body model
class ResearchRequest(BaseModel):
    topic: str

# response body model
class ResearchResponse(BaseModel):
    response: str 

@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    try:
        logger.info(f"Received research request for topic: {request.topic}")
        response = generate_response(request.topic)
        logger.info("Research response generated successfully")
        return ResearchResponse(response=response)
    except Exception as e:
        logger.error(f"Error during research: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
