import json

from django.http import JsonResponse, QueryDict
from matplotlib import pyplot as plt
from rest_framework import serializers
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf

from imdb.imdb_service import ImdbService

@api_view(['POST', 'GET'])
@parser_classes([JSONParser])
def imdb(request):
    if request.method == 'POST':
        id = json.loads(request.body)  # json to dict
        print(f"######## POST id is {id} type is {type(id)} ########")
        a = ImdbService().service_model(int(id))
        print(f" 리턴결과 : {a} ")
        return JsonResponse({'result': a})
    elif request.method == 'GET':
        print(f"######## GET id is {request.GET['id']} ########")
        return JsonResponse(
            {'result': ImdbService().service_model(int(request.GET['id']))})
    else:
        print(f"######## ID is None ########")



#
# from django.http import JsonResponse
# from rest_framework.decorators import api_view, parser_classes
# from rest_framework.parsers import JSONParser
#
# from imdb_service import ImdbService
#
#
# @api_view(['GET'])
# @parser_classes([JSONParser])
# def naver_movie(request):
#     return JsonResponse(
#         {'result': ImdbService().service_model()})
