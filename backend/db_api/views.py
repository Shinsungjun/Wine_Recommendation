from django.http.response import HttpResponse, JsonResponse
from django.shortcuts import render
from db_api.models import Test
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

@method_decorator(csrf_exempt, name='dispatch')
def test(request):
    if request.method == 'GET':
        test = Test.objects.get(dataNumber=0)

        data={
            "dataNumber" : test.dataNumber,
            "name" : test.name,
            "description" : test.description,
            
        }
        return JsonResponse(data)
    else:
        return HttpResponse("Wrong Method!")


@method_decorator(csrf_exempt, name='dispatch')
def recommend(request):
    if request.method == 'POST':
        print(request.body)
        data=request.POST
        # print(json.loads(request.body.decode('utf-8')))
        return JsonResponse(data)
    else:
        return HttpResponse("Wrong Method!")
# Create your views here.
