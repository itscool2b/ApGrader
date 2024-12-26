from django.urls import path
from .views import ApushLEQ, saq_view

urlpatterns = [
    path("ApushLEQ", ApushLEQ, name='ApushLEQ'),
    path("ApushSAQ", saq_view , name='ApushSAQ')
]
