from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import cloudinary.uploader
from .chatbot import gemini_chat, groq_chat, generate_image_from_prompt


@api_view(['POST'])
def chat_with_gemini(request):
    prompt = request.data.get("prompt", "")
    if not prompt:
        return Response({"error": "Prompt required"}, status=status.HTTP_400_BAD_REQUEST)
    response = gemini_chat(prompt)
    return Response({"response": response})


@api_view(['POST'])
def chat_with_groq(request):
    prompt = request.data.get("prompt", "")
    if not prompt:
        return Response({"error": "Prompt Required"}, status=status.HTTP_400_BAD_REQUEST)
    response = groq_chat(prompt)
    return Response({"response": response})


@api_view(['POST'])
def generate_image(request):
    if not isinstance(request.data, dict):
        return Response({"error": "Invalid request format. Use POST with JSON body."}, status=400)

    prompt = request.data.get("prompt", "")
    if not prompt:
        return Response({"error": "Prompt required"}, status=400)

    image = generate_image_from_prompt(prompt)  # see note below
    if isinstance(image, str):
        return Response({"error": image}, status=500)
    
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    upload_result = cloudinary.uploader.upload(buffer, folder="generaged_images")

    image.save("generated_image.png")
    return Response({"image_url": upload_result['secure_url']})

