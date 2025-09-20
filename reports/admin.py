from django.contrib import admin
from django.contrib.gis.admin import GISModelAdmin
from .models import Reported, CleanedReport

@admin.register(Reported)
class ReportedAdmin(GISModelAdmin):
    """Admin configuration for the Reported model."""
    list_display = ('id', 'user', 'reported_at', 'is_marked', 'source')
    list_filter = ('is_marked', 'reported_at', 'source')
    search_fields = ('user__email', 'report')
    readonly_fields = ('reported_at',)
    
    # Map widget settings
    default_lat = 20
    default_lon = 78
    default_zoom = 4


@admin.register(CleanedReport)
class CleanedReportAdmin(GISModelAdmin):
    """Admin configuration for the CleanedReport model."""
    list_display = ('id', 'get_original_report_id', 'hazard_type', 'severity', 'confidence', 'verified', 'processed_at')
    list_filter = ('hazard_type', 'severity', 'verified', 'language')
    search_fields = ('cleaned_text', 'original_report__report')
    readonly_fields = ('processed_at',)

    def get_original_report_id(self, obj):
        """Method to display the related original report ID in the list view."""
        return obj.original_report.id
    
    get_original_report_id.short_description = 'Original Report ID'
    get_original_report_id.admin_order_field = 'original_report__id'