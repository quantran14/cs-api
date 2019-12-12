import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Create your views here.


def upload_images(request):
    print(request.FILES)
    img = request.FILES.get('image')

    file_saving = FileSystemStorage(settings.MEDIA_ROOT, settings.MEDIA_URL)
    file_saving.save(img.name, img)




class Image(APIView):
    print(1)

    def post(self, request, *args, **kwargs):
        upload_images(request=request)

        return Response({'success': 'accepted'}, status= status.HTTP_202_ACCEPTED)
