from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from dotenv import load_dotenv
from app.utils.logger import logger
import os

# Load env
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")

# =========================
# TEXT LLM
# =========================
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    logger.success("LLM initialized")

except Exception as e:
    logger.error(f"LLM init failed: {e}")
    raise


# =========================
# IMAGE CLIENT (RAW)
# =========================
try:
    image_client = genai.Client(api_key=API_KEY)
    logger.success("Image client initialized")

except Exception as e:
    logger.error(f"Image client init failed: {e}")
    raise


# =========================
# IMAGE GENERATION FUNCTION
# =========================
def generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes (PNG)
    """
    try:
        response = image_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt
        )

        # Extract image bytes
        parts = getattr(response, "parts", None)

        if not parts and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts

        if not parts:
            raise RuntimeError("No image parts returned")

        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data:
                return part.inline_data.data

        raise RuntimeError("No image data found")

    except Exception as e:
        logger.error({
            "event": "image_generation_failed",
            "error": str(e),
            "prompt": prompt
        })
        raise