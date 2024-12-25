from django.urls import path
from .views import ApushLEQ

urlpatterns = [
    path("ApushLEQ", ApushLEQ, name='ApushLEQ'),
    
]
