from django.urls import path
from .views import process
from .views import process_prompt
urlpatterns = [
    path("essay", process, name='process'),
    path("prompt", process_prompt, name='process_prompt')
]
