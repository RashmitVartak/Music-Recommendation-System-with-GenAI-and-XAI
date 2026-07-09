from django.contrib import admin
from django.urls import path
from SE_miniproject.miniproject.UI import views

urlpatterns = [
    path('', views.index,name='UI')
]
