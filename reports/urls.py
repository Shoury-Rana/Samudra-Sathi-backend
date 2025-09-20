from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ReportedViewSet, CleanedReportViewSet

router = DefaultRouter()
router.register(r'incidents', ReportedViewSet, basename='reported')
router.register(r'cleaned-incidents', CleanedReportViewSet, basename='cleanedreport')

urlpatterns = [
    path('', include(router.urls)),
]