from django.shortcuts import render, HttpResponse

# Create your views here.
def index(reuqest):
    return HttpResponse("This is MY home page")