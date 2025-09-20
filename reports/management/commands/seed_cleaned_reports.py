import random
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point, GeometryCollection
from django.db import transaction
from django.utils import timezone
from faker import Faker
from users.models import User
from reports.models import Reported, CleanedReport

class Command(BaseCommand):
    help = 'Seeds the database with pairs of raw and cleaned reports'

    def add_arguments(self, parser):
        parser.add_argument('--number', type=int, help='The number of report pairs to create', default=500)

    @transaction.atomic
    def handle(self, *args, **options):
        fake = Faker('en_IN')
        number_of_reports = options['number']

        users = list(User.objects.all())
        if not users:
            self.stdout.write(self.style.ERROR('No users found. Please run `seed_users` first.'))
            return

        # Bounding box for India
        lat_min, lat_max = 8.0, 37.0
        lon_min, lon_max = 68.0, 97.0
        
        hazard_types = [choice[0] for choice in CleanedReport.HazardType.choices]
        severities = [choice[0] for choice in CleanedReport.Severity.choices]
        sentiments = ['positive', 'negative', 'neutral']
        sources = ['Web', 'Mobile App', 'SMS', 'API']
        
        self.stdout.write(f'Creating {number_of_reports} new raw and cleaned report pairs...')

        for _ in range(number_of_reports):
            # --- 1. Create a raw Reported instance ---
            latitude = random.uniform(lat_min, lat_max)
            longitude = random.uniform(lon_min, lon_max)
            location_point = Point(longitude, latitude, srid=4326)
            
            raw_report = Reported.objects.create(
                user=random.choice(users),
                report=f"Emergency near {fake.city()}. {fake.sentence()}",
                is_marked=True, # Mark as processed since we are creating a cleaned version
                location=location_point,
                source=random.choice(sources)
            )

            # --- 2. Create the associated CleanedReport ---
            CleanedReport.objects.create(
                original_report=raw_report,
                cleaned_text=raw_report.report, # For simplicity, use the same text
                timestamp=fake.date_time_between(start_date='-1y', end_date='now', tzinfo=timezone.get_current_timezone()),
                language='en',
                hazard_type=random.choice(hazard_types),
                confidence=round(random.uniform(0.75, 1.0), 2),
                severity=random.choice(severities),
                locations=GeometryCollection(location_point), # Wrap the point in a collection
                sentiment=random.choice(sentiments),
                urgency_score=round(random.uniform(0.1, 1.0), 2),
                source=raw_report.source,
                verified=random.choice([True, False])
            )

        self.stdout.write(self.style.SUCCESS(f'Successfully created {number_of_reports} raw and cleaned report pairs.'))