import random
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point, GeometryCollection
from django.db import transaction
from django.utils import timezone
from faker import Faker
from users.models import User
from reports.models import Reported, CleanedReport

# Use the same realistic hotspot definitions and templates
# (In a real project, this could be refactored into a shared utility)
HOTSPOTS = {
    "coastal": [
        ("Mumbai", (19.0760, 72.8777), ["FLOOD", "HIGH_WAVES", "STORM_SURGE"]),
        ("Chennai", (13.0827, 80.2707), ["TSUNAMI", "FLOOD", "CYCLONE"]),
        ("Kolkata", (22.5726, 88.3639), ["CYCLONE", "FLOOD", "STORM_SURGE"]),
        ("Kochi", (9.9312, 76.2673), ["FLOOD", "HIGH_WAVES"]),
        ("Vizag", (17.6868, 83.2185), ["CYCLONE", "TSUNAMI"]),
        ("Puri", (19.8135, 85.8312), ["CYCLONE", "STORM_SURGE"]),
    ],
    "himalayan": [
        ("Joshimath", (30.5656, 79.5632), ["LANDSLIDE", "EARTHQUAKE"]),
        ("Shimla", (31.1048, 77.1734), ["EARTHQUAKE", "LANDSLIDE"]),
    ],
    "inland": [
        ("Delhi", (28.7041, 77.1025), ["FLOOD", "EARTHQUAKE"]),
        ("Bangalore", (12.9716, 77.5946), ["FLOOD"]),
    ]
}

REPORT_TEMPLATES = {
    "FLOOD": ["High water levels near {}.", "Urban flooding in {}.", "River is overflowing its banks in {} area."],
    "TSUNAMI": ["The sea is receding unusually at {}.", "Massive waves hitting the coast at {}."],
    "EARTHQUAKE": ["The ground is shaking violently in {}.", "My building in {} is swaying."],
    "LANDSLIDE": ["A huge landslide has blocked the road to {}.", "Mudslide reported in the hills above {}."],
    "CYCLONE": ["Cyclone warning for {}.", "The wind is howling in {} due to the storm."],
    # Simplified for brevity
}


class Command(BaseCommand):
    help = 'Seeds the database with pairs of realistic raw and cleaned reports.'

    def add_arguments(self, parser):
        parser.add_argument('--number', type=int, help='The number of report pairs to create', default=200)

    @transaction.atomic
    def handle(self, *args, **options):
        fake = Faker('en_IN')
        number_of_reports = options['number']

        users = list(User.objects.all())
        if not users:
            self.stdout.write(self.style.ERROR('No users found. Please run `seed_users500` first.'))
            return

        all_hotspots = HOTSPOTS["coastal"] + HOTSPOTS["himalayan"] + HOTSPOTS["inland"]
        severities = [choice[0] for choice in CleanedReport.Severity.choices]
        sentiments = ['positive', 'negative', 'neutral']
        sources = ['Web', 'Mobile App', 'SMS', 'API']
        
        self.stdout.write(f'Creating {number_of_reports} new raw and cleaned report pairs...')

        for _ in range(number_of_reports):
            # --- 1. Create a realistic raw Reported instance ---
            hotspot_name, (lat, lon), hazard_types = random.choice(all_hotspots)
            hazard_type_for_raw_report = random.choice(hazard_types)

            latitude = random.gauss(lat, 0.1)
            longitude = random.gauss(lon, 0.1)
            location_point = Point(longitude, latitude, srid=4326)
            
            template = random.choice(REPORT_TEMPLATES.get(hazard_type_for_raw_report, REPORT_TEMPLATES["FLOOD"]))
            report_text = template.format(hotspot_name)

            raw_report = Reported.objects.create(
                user=random.choice(users),
                report=report_text,
                is_marked=True,  # Mark as processed
                location=location_point,
                source=random.choice(sources)
            )

            # --- 2. Create a consistent, cleaned report ---
            # The AI "correctly" identifies a hazard type applicable to the location
            cleaned_hazard_type = random.choice(hazard_types)

            CleanedReport.objects.create(
                original_report=raw_report,
                cleaned_text=f"Confirmed {cleaned_hazard_type.lower()} event near {hotspot_name}.",
                timestamp=fake.date_time_between(start_date='-1y', end_date='now', tzinfo=timezone.get_current_timezone()),
                language='en',
                hazard_type=cleaned_hazard_type,
                confidence=round(random.uniform(0.85, 1.0), 2),
                severity=random.choice(severities),
                locations=GeometryCollection(location_point),
                sentiment=random.choice(sentiments),
                urgency_score=round(random.uniform(0.5, 1.0), 2),
                source=raw_report.source,
                verified=random.choice([True, False, False]) # Skew towards unverified
            )

        self.stdout.write(self.style.SUCCESS(f'Successfully created {number_of_reports} realistic report pairs.'))