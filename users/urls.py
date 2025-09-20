from django.urls import path
from .views import (
    UserRegisterView, 
    LogoutView, 
    UserLocationsView, 
    UserDensityView
)
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns = [
    path('register/', UserRegisterView.as_view(), name='register'),
    path('login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('login/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', LogoutView.as_view(), name='logout'),
    
    path('locations/', UserLocationsView.as_view(), name='user-locations'),
    path('locations/density/', UserDensityView.as_view(), name='user-density'),
]
