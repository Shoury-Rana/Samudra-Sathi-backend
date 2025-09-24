import requests
import json
import math
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import transaction
from django.contrib.gis.geos import Point, GeometryCollection
from reports.models import Reported, CleanedReport

class Command(BaseCommand):
    help = 'Interactively fetches and processes raw reports in user-defined batches.'

    @transaction.atomic
    def handle(self, *args, **options):
        # --- Step 1: Check for available work ---
        unprocessed_reports = Reported.objects.filter(is_marked=False).exclude(report__exact='').exclude(report__isnull=True)
        report_count = unprocessed_reports.count()
        
        if report_count == 0:
            self.stdout.write(self.style.SUCCESS('No new valid raw reports to process.'))
            return

        self.stdout.write(self.style.SUCCESS(f'Found {report_count} new reports available to process.'))

        # --- Step 2: Get interactive input from the user ---
        try:
            # Ask for number of batches
            while True:
                num_batches_str = input("How many batches would you like to process? (e.g., 2): ")
                try:
                    num_batches = int(num_batches_str)
                    if num_batches > 0:
                        break
                    else:
                        self.stdout.write(self.style.WARNING("Please enter a number greater than 0."))
                except ValueError:
                    self.stdout.write(self.style.ERROR("Invalid input. Please enter a whole number."))

            # Ask for reports per batch
            while True:
                batch_size_str = input("How many reports per batch? (Max 100, e.g., 50): ")
                try:
                    batch_size = int(batch_size_str)
                    if 0 < batch_size <= 100:
                        break
                    else:
                        self.stdout.write(self.style.WARNING("Please enter a number between 1 and 100."))
                except ValueError:
                    self.stdout.write(self.style.ERROR("Invalid input. Please enter a whole number."))
        
        except KeyboardInterrupt:
            self.stdout.write(self.style.ERROR("\nOperation cancelled by user."))
            return


        # --- Step 3: Calculate and confirm the processing plan ---
        total_to_process_requested = num_batches * batch_size
        # Make sure we don't try to process more reports than are available
        actual_to_process = min(total_to_process_requested, report_count)
        actual_batches = math.ceil(actual_to_process / batch_size)

        self.stdout.write(self.style.SUCCESS(f'\nPlan: Processing {actual_to_process} reports in {actual_batches} batch(es) of up to {batch_size}.'))
        
        # --- Step 4: Execute the plan ---
        reports_to_process = unprocessed_reports[:actual_to_process]
        all_cleaned_reports_to_create = []
        all_reports_to_update_as_marked = []

        for i in range(0, actual_to_process, batch_size):
            batch_num = (i // batch_size) + 1
            self.stdout.write(self.style.HTTP_INFO(f'--- Processing batch {batch_num} of {actual_batches} ---'))
            
            batch = reports_to_process[i:i + batch_size]
            
            payload = [{"text": report.report, "source": report.source or "Unknown"} for report in batch]
            report_map = {report.report: report for report in batch}

            try:
                self.stdout.write(f'Sending {len(payload)} reports to the AI service...')
                response = requests.post(settings.AI_SERVICE_URL, json={"reports": payload}, timeout=120)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                self.stdout.write(self.style.ERROR(f'Failed to process batch {batch_num}: {e}'))
                if hasattr(e, 'response') and e.response is not None:
                    self.stdout.write(self.style.ERROR(f'AI Service Response: {e.response.text}'))
                self.stdout.write(self.style.ERROR('Aborting operation due to batch failure.'))
                return

            ai_results = response.json()
            for result in ai_results:
                original_report = report_map.get(result['text'])
                if not original_report or original_report.is_marked:
                    continue

                geometries = [Point(loc['longitude'], loc['latitude'], srid=4326) for loc in result.get('locations', []) if loc.get('latitude') is not None]
                if not geometries and original_report.location:
                    geometries.append(original_report.location)
                if not geometries: continue

                cleaned_report_data = {
                    'original_report': original_report, 'cleaned_text': result['text'], 'timestamp': result['timestamp'],
                    'language': result['language'], 'hazard_type': result['hazard_type'].upper(), 'confidence': result['confidence'],
                    'severity': result['severity'].upper(), 'locations': GeometryCollection(geometries), 'sentiment': result['sentiment'],
                    'urgency_score': result['urgency_score'], 'source': result['source'], 'verified': result.get('verified', False)
                }
                all_cleaned_reports_to_create.append(CleanedReport(**cleaned_report_data))
                original_report.is_marked = True
                all_reports_to_update_as_marked.append(original_report)

        # --- Step 5: Save results to the database ---
        if all_cleaned_reports_to_create:
            CleanedReport.objects.bulk_create(all_cleaned_reports_to_create, ignore_conflicts=True)
            self.stdout.write(self.style.SUCCESS(f'Successfully created {len(all_cleaned_reports_to_create)} new cleaned reports.'))
        if all_reports_to_update_as_marked:
            Reported.objects.bulk_update(all_reports_to_update_as_marked, ['is_marked'])
            self.stdout.write(self.style.SUCCESS(f'Successfully marked {len(all_reports_to_update_as_marked)} raw reports as processed.'))