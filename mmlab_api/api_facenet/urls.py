from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import url
from .views import Image

urlpatterns = [
    url(r'^image/$', Image.as_view(), name='Image'),
]

urlpatterns += static(settings.MEDIA_URL,
                      document_root=settings.MEDIA_ROOT_FACENET)
