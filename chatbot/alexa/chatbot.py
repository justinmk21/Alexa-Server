from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import InferenceClient
from PIL import Image
from groq import Groq
import os
import io

load_dotenv()

# Load API Key from Google AI Studio
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load API Key for stable Diffusion Image Generator.
client = InferenceClient(
    provider="nscale",
    api_key=os.environ["HF_TOKEN"],
)

headers = {"Authorization": f"Bearer {os.getenv("HF_TOKEN")}"}

def gemini_chat(prompt: str) -> str:
    """
    Chat with Google's Gemini LLM.
    """
    try:
        response = model.generate_content(
            f"Respond to the following prompt clearly and concisely (max 300 words):\n\n{prompt}"
        )
        # Extract text safely
        if hasattr(response, "text"):
            return response.text
        else:
            return "No text field found in response."
    except Exception as e:
        return f"Error in Gemini chat: {e}"


def generate_image_from_prompt(prompt: str) -> str:
    """
    Getting Images generated from Stable Difussion.
    """
    try:
        result = client.text_to_image(prompt, model="stabilityai/stable-diffusion-xl-base-1.0")

        if isinstance(result, bytes):
            image = Image.open(io.BytesIO(result))
            return image
        elif isinstance(result, Image.Image):
            return result
        else:
            return f"Unexpected output type: {type(result)}"
    except Exception as e:
        return f"Error generating image: {e}"


def groq_chat(prompt: str) -> str:
    """
    Chat with Groq Llama-3 or Mixtral models.
    """
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a creative AI assistant chatbot."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq error: {e}"
