from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from app.utils.logger import logger

# Load env
load_dotenv()
logger.info("Environment variables loaded")

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    logger.success("LLM initialized successfully: gemini-2.5-flash")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise