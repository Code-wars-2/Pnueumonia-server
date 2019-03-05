from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
from utils  import xrayclassification2
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
@api_view(["POST"])
def pneumonia(imagedata):
   image=json.loads(imagedata.body.decode())
   var= str(xrayclassification2.model(image))
   return HttpResponse(var)
   	  


