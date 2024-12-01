from django.urls import path
from .views import process


urlpatterns = [
    path("sa/api/pdf", process, name='process'),
]