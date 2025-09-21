from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ReportedViewSet, 
    CleanedReportViewSet,
    ReportedLocationsView,
    ReportedDensityView,
    CleanedReportLocationsView,
    CleanedReportDensityView
)

router = DefaultRouter()
router.register(r'incidents', ReportedViewSet, basename='reported')
router.register(r'cleaned-incidents', CleanedReportViewSet, basename='cleanedreport')

urlpatterns = [
    path('incidents/locations/', ReportedLocationsView.as_view(), name='reported-locations'),
    path('incidents/density/', ReportedDensityView.as_view(), name='reported-density'),
    
    path('cleaned-incidents/locations/', CleanedReportLocationsView.as_view(), name='cleanedreport-locations'),
    path('cleaned-incidents/density/', CleanedReportDensityView.as_view(), name='cleanedreport-density'),

    path('', include(router.urls)),
]
