from django.urls import path
from . import views

urlpatterns = [
    path("ApEuroDBQ", views.eurodbq, name='ApEuroDBQ'),
    path("ApushLEQ", views.ApushLEQ, name='ApushLEQ'),
    path("ApushSAQ", views.saq_view , name='ApushSAQ'),
    path("ApushDBQ", views.dbq_view, name='ApushDBQ'),
    path("ApEuroLEQ", views.ApEuroLEQ, name='ApEuroLEQ'),
    path("ApEuroSAQ", views.eurosaq_view, name='ApEuroSAQ'),
    
]
