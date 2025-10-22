from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views

urlpatterns = [
    path('gemini/', views.chat_with_gemini, name='chat_gemini'),
    path('groq/', views.chat_with_groq, name='chat_groq'),
    path('image/', views.generate_image, name='image_gen'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
