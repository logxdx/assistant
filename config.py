import os

from dotenv import load_dotenv

load_dotenv()

# TTS Parameters
ASSISTANT_NAME = "Jarvis"

Assistants = {
    "jarvis": 
    {
        "name": "Jarvis",
        "keyword": "hey jarvis",
        "oww_path": r"C:\Code\AI\assistant\models\oww\hey_jarvis.onnx", 
        "tts_config": {
            "voice": "am_onyx(1)+am_fenrir(1)",
            "speed": "1.25",
        }
    },

    "friday": 
    {
        "name": "Friday",
        "keyword": "friday",
        "oww_path": r"C:\Code\AI\assistant\models\oww\FRYDAY.onnx",
        "tts_config": {
            "voice": "af_heart",
            "speed": "1.25",
        }
    }
}

# Function to get the assistant model
def get_assistant_model(assistant_name: str = ASSISTANT_NAME) -> dict:
    """
    Get the assistant model based on the assistant name.
    Args:
        assistant_name (str): The name of the assistant.
        - Jarvis
        - qwen2.5
    Returns:
        tuple: A tuple containing the wake word, wake word model path, and the speaker settings.
    """
    return Assistants[assistant_name.lower()]


# Model API Params

# General
GENERAL_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
GENERAL_API_KEY = os.getenv("CEREBRAS_API_KEY")
GENERAL_MODEL = "llama-3.3-70b"

# Decision
DECISION_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
DECISION_API_KEY = os.getenv("CEREBRAS_API_KEY")
DECISION_MODEL = "llama-3.3-70b"

# Tool use
TOOL_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
TOOL_API_KEY = os.getenv("CEREBRAS_API_KEY")
TOOL_MODEL = "llama-3.3-70b"

# Web Scraper
# SCRAPER_PROVIDER = 'cerebras'
# SCRAPER_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
# SCRAPER_API_KEY = os.getenv("CEREBRAS_API_KEY")
# SCRAPER_MODEL = "llama3.1-8b"
SCRAPER_PROVIDER = 'openrouter'
SCRAPER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
SCRAPER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SCRAPER_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"
SEARXNG_URL = os.getenv("SEARXNG_URL")

# Summarisation
# SUMMARISATION_BASE_URL = os.getenv("CEREBRAS_BASE_URL")
# SUMMARISATION_API_KEY = os.getenv("CEREBRAS_API_KEY")
# SUMMARISATION_MODEL = "llama3.1-8b"
SUMMARISATION_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
SUMMARISATION_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUMMARISATION_MODEL = "google/gemini-2.0-flash-lite-preview-02-05:free"

# Vision
# VISION_BASE_URL = os.getenv("GROQ_BASE_URL")
# VISION_API_KEY = os.getenv("GROQ_API_KEY")
# VISION_MODEL = "llama-3.2-90b-vision-preview"
VISION_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
VISION_API_KEY = os.getenv("OPENROUTER_API_KEY")
VISION_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

# Code
CODE_BASE_URL = os.getenv("GROQ_BASE_URL")
CODE_API_KEY = os.getenv("GROQ_API_KEY")
CODE_MODEL = "qwen-2.5-coder-32b"
