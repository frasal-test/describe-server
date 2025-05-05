from dotenv import load_dotenv
import os

load_dotenv()

#API_KEY = os.getenv("API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
HF_TOKEN = os.getenv("HF_TOKEN")
