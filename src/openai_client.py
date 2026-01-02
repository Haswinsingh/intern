from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()   # 🔥 VERY IMPORTANT

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI()
