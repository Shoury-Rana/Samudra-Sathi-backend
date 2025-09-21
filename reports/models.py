from django.conf import settings
from django.contrib.gis.db import models
from django.utils import timezone

class Reported(models.Model):
    """
    Model to store raw reports submitted by users.
    """
    report = models.TextField()
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='reports', null=True, blank=True)
    is_marked = models.BooleanField(default=False)
    reported_at = models.DateTimeField(default=timezone.now)
    location = models.PointField(srid=4326, null=True, blank=True)
    source = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return f"Report by {self.user.email} at {self.reported_at.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ['-reported_at']


class CleanedReport(models.Model):
    """
    Model to store the structured data after an AI has processed a Reported instance.
    """
    class Severity(models.TextChoices):
        LOW = 'LOW', 'Low'
        MEDIUM = 'MEDIUM', 'Medium'
        HIGH = 'HIGH', 'High'

    class HazardType(models.TextChoices):
        FLOOD  = 'FLOOD', 'flood'
        TSUNAMI = 'TSUNAMI', 'tsunami'
        EARTHQUAKE = 'EARTHQUAKE', 'earthquake'
        OTHER = 'OTHER', 'Other'

    original_report = models.OneToOneField(Reported, on_delete=models.CASCADE, related_name='cleaned_report')
    cleaned_text = models.TextField()
    timestamp = models.DateTimeField()
    language = models.CharField(max_length=50)
    hazard_type = models.CharField(max_length=50, choices=HazardType.choices, default=HazardType.OTHER)
    confidence = models.FloatField()
    severity = models.CharField(max_length=10, choices=Severity.choices, default=Severity.LOW)
    locations = models.GeometryCollectionField(srid=4326)
    sentiment = models.CharField(max_length=20)
    urgency_score = models.FloatField()
    source = models.CharField(max_length=50, blank=True)
    verified = models.BooleanField(default=False)
    processed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Cleaned report for ID {self.original_report.id}"

    class Meta:
        ordering = ['-processed_at']