from rest_framework import viewsets, permissions
from .models import Reported, CleanedReport
from .serializers import ReportedSerializer, CleanedReportSerializer

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