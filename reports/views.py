from rest_framework import viewsets, permissions, generics, views, status
from rest_framework.response import Response
from rest_framework_gis.filters import InBBoxFilter
from django.db import connection
import json

from .models import Reported, CleanedReport
from .serializers import (
    ReportedSerializer, 
    CleanedReportSerializer,
    ReportedLocationSerializer,
    CleanedReportLocationSerializer
)


class ReportedViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to create, view, edit, and delete their own reports.
    """
    serializer_class = ReportedSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """
        This view should return a list of all the reports
        for the currently authenticated user.
        """
        return Reported.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """
        Assign the current user to the report being created.
        """
        serializer.save(user=self.request.user)


class CleanedReportViewSet(viewsets.ModelViewSet):
    """
    API endpoint for CRUD operations on AI-cleaned reports.
    Typically, these would be created by a backend process, but this
    endpoint allows for manual creation and management.
    """
    queryset = CleanedReport.objects.all()
    serializer_class = CleanedReportSerializer
    permission_classes = [permissions.IsAuthenticated]


class BaseDensityView(views.APIView):
    """Base class for heatmap density views to avoid code duplication."""
    permission_classes = [permissions.IsAuthenticated]
    table_name = None
    geom_field = None

    def get_grid_size(self, zoom):
        """Determines grid cell size in degrees based on map zoom level."""
        if zoom < 3: return 5.0
        elif zoom < 5: return 2.0
        elif zoom < 7: return 0.5
        elif zoom < 9: return 0.1
        elif zoom < 12: return 0.02
        else: return 0.005

    def get_geom_expression(self):
        """Returns the geometry expression for the SQL query."""
        return self.geom_field

    def get(self, request, *args, **kwargs):
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

        grid_size = self.get_grid_size(zoom)
        geom_expression = self.get_geom_expression()

        with connection.cursor() as cursor:
            query = f"""
                SELECT
                    COUNT(*) as count,
                    ST_AsGeoJSON(ST_SnapToGrid({geom_expression}, %(grid_size)s, %(grid_size)s)) as point_geojson
                FROM
                    {self.table_name}
                WHERE
                    {self.geom_field} IS NOT NULL AND
                    {self.geom_field} && ST_MakeEnvelope(%(xmin)s, %(ymin)s, %(xmax)s, %(ymax)s, 4326)
                GROUP BY
                    ST_SnapToGrid({geom_expression}, %(grid_size)s, %(grid_size)s);
            """
            params = {'grid_size': grid_size, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
            cursor.execute(query, params)
            rows = cursor.fetchall()

        features = []
        for count, point_geojson in rows:
            if point_geojson:
                features.append({
                    "type": "Feature",
                    "geometry": json.loads(point_geojson),
                    "properties": {"count": count}
                })

        return Response({"type": "FeatureCollection", "features": features})


class ReportedLocationsView(generics.ListAPIView):
    """
    Returns a GeoJSON FeatureCollection of raw reports with locations.
    Filterable by bounding box: `?in_bbox=<xmin>,<ymin>,<xmax>,<ymax>`
    """
    queryset = Reported.objects.filter(location__isnull=False)
    serializer_class = ReportedLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = (InBBoxFilter,)
    bbox_filter_field = 'location'
    pagination_class = None


class ReportedDensityView(BaseDensityView):
    """
    Returns a GeoJSON FeatureCollection for a raw report density heatmap.
    Query parameters: `in_bbox` and `zoom`.
    """
    table_name = 'reports_reported'
    geom_field = 'location'


class CleanedReportLocationsView(generics.ListAPIView):
    """
    Returns a GeoJSON FeatureCollection of cleaned reports with locations.
    Filterable by bounding box: `?in_bbox=<xmin>,<ymin>,<xmax>,<ymax>`
    """
    queryset = CleanedReport.objects.filter(locations__isnull=False)
    serializer_class = CleanedReportLocationSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = (InBBoxFilter,)
    bbox_filter_field = 'locations'
    pagination_class = None


class CleanedReportDensityView(BaseDensityView):
    """
    Returns a GeoJSON FeatureCollection for a cleaned report density heatmap.
    Query parameters: `in_bbox` and `zoom`.
    """
    table_name = 'reports_cleanedreport'
    geom_field = 'locations'

    def get_geom_expression(self):
        # Use the centroid of the geometry collection for heatmap aggregation.
        return f'ST_Centroid({self.geom_field})'