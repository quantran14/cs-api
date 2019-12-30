from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import url
from .views import ImageDetector, ImageExtractor

urlpatterns = [
    url(r'^detect/$', ImageDetector.as_view(), name='ImageDetector'),
    url(r'^extract/$', ImageExtractor.as_view(), name='ImageExtractor'),
]

urlpatterns += static(settings.MEDIA_URL,
                      document_root=settings.MEDIA_ROOT_VGGFACE)
