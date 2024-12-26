from django.urls import path
from . import views

urlpatterns = [
    path("ApushLEQ", views.ApushLEQ, name='ApushLEQ'),
    path("ApushSAQ", views.saq_view , name='ApushSAQ')
]
