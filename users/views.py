from rest_framework import generics, status, views
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import UserRegisterSerializer, LogoutSerializer, UserSerializer, UserLocationSerializer
from .models import User
from rest_framework_gis.filters import InBBoxFilter

from django.db import connection
import json


class UserRegisterView(generics.GenericAPIView):
    serializer_class = UserRegisterSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Use UserSerializer to properly convert location to dict
        user_data = UserSerializer(user).data

        return Response({
            "user": user_data,
            "message": "User Created Successfully. Now perform Login to get your token.",
        }, status=status.HTTP_201_CREATED)


class LogoutView(generics.GenericAPIView):
    serializer_class = LogoutSerializer
    permission_classes = (AllowAny,)

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response({"detail": "Successfully logged out"}, status=status.HTTP_200_OK)


class UserLocationsView(generics.ListAPIView):
    """
    Returns a GeoJSON FeatureCollection of users with locations.
    
    This endpoint is filterable by a bounding box using the `in_bbox`
    query parameter. The format is `in_bbox=<xmin>,<ymin>,<xmax>,<ymax>`.

    Example: `/api/users/locations/?in_bbox=70,15,90,30`
    """
    queryset = User.objects.filter(location__isnull=False)
    serializer_class = UserLocationSerializer
    permission_classes = [AllowAny] 
    
    # Use the InBBoxFilter for geospatial filtering
    filter_backends = (InBBoxFilter,)
    
    # Specify the geometry field to filter on
    bbox_filter_field = 'location'

    # Optional: Disable pagination for GeoJSON responses
    pagination_class = None


class UserDensityView(views.APIView):
    """
    Returns a GeoJSON FeatureCollection for generating a user density heatmap.

    Aggregates user locations into a grid and returns the count of users
    in each grid cell. The grid size is determined by the zoom level.

    **Query Parameters:**

    - `in_bbox` (required): The bounding box of the map view.
      Format: `<xmin>,<ymin>,<xmax>,<ymax>`
      Example: `/api/users/locations/density/?in_bbox=70,15,90,30&zoom=5`

    - `zoom` (required): The current zoom level of the map (integer).
    """
    permission_classes = [AllowAny] 

    def get_grid_size(self, zoom):
        """
        Determines the grid cell size in degrees based on the map zoom level.
        This is a simple implementation; a more complex logarithmic scale
        could be used for smoother transitions.
        """
        if zoom < 3:
            return 5.0  # Very large cells for world view.      Approx. 40k km2.
        elif zoom < 5:
            return 2.0
        elif zoom < 7:
            return 0.5
        elif zoom < 9:
            return 0.1
        elif zoom < 12:
            return 0.02
        else:
            return 0.005 # Small cells for city-level view.     Approx. 40 km2.

    def get(self, request, *args, **kwargs):
        # 1. Parse and validate query parameters
        bbox_str = request.query_params.get('in_bbox')
        zoom_str = request.query_params.get('zoom')

        if not bbox_str or not zoom_str:
            return Response(
                {"error": "Missing required query parameters: 'in_bbox' and 'zoom'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            zoom = int(zoom_str)
            xmin, ymin, xmax, ymax = [float(c) for c in bbox_str.split(',')]
        except (ValueError, TypeError):
            return Response(
                {"error": "Invalid format for 'in_bbox' or 'zoom'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Determine grid size
        grid_size = self.get_grid_size(zoom)

        # 3. Execute the raw SQL query for aggregation
        with connection.cursor() as cursor:
            # Using query parameters (%(...)s) to prevent SQL injection
            query = """
                SELECT
                    COUNT(*) as count,
                    ST_AsGeoJSON(ST_SnapToGrid(location, %(grid_size)s, %(grid_size)s)) as point_geojson
                FROM
                    users_user
                WHERE
                    location IS NOT NULL AND
                    location && ST_MakeEnvelope(%(xmin)s, %(ymin)s, %(xmax)s, %(ymax)s, 4326)
                GROUP BY
                    ST_SnapToGrid(location, %(grid_size)s, %(grid_size)s);
            """
            params = {
                'grid_size': grid_size,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }
            cursor.execute(query, params)
            rows = cursor.fetchall()

        # 4. Format the results into a GeoJSON FeatureCollection
        features = []
        for row in rows:
            count, point_geojson = row
            if point_geojson:
                features.append({
                    "type": "Feature",
                    "geometry": json.loads(point_geojson),
                    "properties": {
                        "count": count
                    }
                })

        geojson_response = {
            "type": "FeatureCollection",
            "features": features
        }

        return Response(geojson_response)