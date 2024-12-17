from django.urls import path
from .views import process

urlpatterns = [
    path("process", process, name='process'),
    
]
