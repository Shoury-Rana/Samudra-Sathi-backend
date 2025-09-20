from rest_framework import serializers
from rest_framework_gis.serializers import GeoFeatureModelSerializer
from django.contrib.gis.geos import Point
from .models import Reported, CleanedReport

class ReportedSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.email')
    location = serializers.DictField(required=False, write_only=True)
    location_details = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Reported
        fields = ['id', 'report', 'user', 'is_marked', 'reported_at', 'location', 'location_details', 'source']
        read_only_fields = ['is_marked', 'reported_at']

    def get_location_details(self, obj):
        if obj.location:
            return {"lat": obj.location.y, "lng": obj.location.x}
        return None

    def create(self, validated_data):
        location_data = validated_data.pop('location', None)
        report = Reported.objects.create(**validated_data)

        if location_data:
            try:
                report.location = Point(location_data['lng'], location_data['lat'], srid=4326)
            except (KeyError, TypeError):
                raise serializers.ValidationError("Invalid location format. Expected {'lat': ..., 'lng': ...}")
        elif report.user.location:
            report.location = report.user.location
        
        report.save()
        return report

class CleanedReportSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = CleanedReport
        geo_field = 'locations'
        fields = '__all__'

# --- GeoJSON Serializers for Map Views ---

class ReportedLocationSerializer(GeoFeatureModelSerializer):
    """
    A lean serializer to convert Reported model instances into GeoJSON Feature objects.
    """
    class Meta:
        model = Reported
        geo_field = "location"
        fields = ('id', 'source')

class CleanedReportLocationSerializer(GeoFeatureModelSerializer):
    """
    A lean serializer to convert CleanedReport model instances into GeoJSON Feature objects.
    """
    class Meta:
        model = CleanedReport
        geo_field = "locations"
        fields = ('id', 'hazard_type', 'severity', 'verified')