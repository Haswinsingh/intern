import os
import google.generativeai as genai

print("🔥 Starting Gemini test")

# API KEY
os.environ["GOOGLE_API_KEY"] = "AIzaSyAVwJmxLJRS9NjlK4pYIGaP-ccaRzI72r8"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ HARD FIXED FREE MODEL
MODEL_NAME = "models/gemini-flash-latest"

print("✅ MODEL SET TO:", MODEL_NAME)

model = genai.GenerativeModel(MODEL_NAME)

response = model.generate_content("Say OK in one word")

print("\n🤖 RESPONSE:")
print(response.text)
