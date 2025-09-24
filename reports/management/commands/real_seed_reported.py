import random
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from faker import Faker
from users.models import User
from reports.models import Reported

# Define realistic hotspots for report clustering
# Format: (Name, (Lat, Lon), [Applicable Hazard Types])
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
        ("Joshimath", (30.5656, 79.5632), ["LANDSLIDE", "FLASH_FLOOD", "EARTHQUAKE"]),
        ("Shimla", (31.1048, 77.1734), ["EARTHQUAKE", "LANDSLIDE"]),
        ("Darjeeling", (27.0360, 88.2627), ["LANDSLIDE"]),
    ],
    "inland": [
        ("Delhi", (28.7041, 77.1025), ["FLOOD", "EARTHQUAKE"]),
        ("Bangalore", (12.9716, 77.5946), ["FLOOD"]), # Urban flooding
        ("Patna", (25.5941, 85.1376), ["FLOOD"]), # River flooding
    ]
}

# Report templates mapped to hazard types
REPORT_TEMPLATES = {
    "FLOOD": [
        "Seeing high water levels near {}. It's rising fast!",
        "Massive urban flooding in {}. Cars are submerged.",
        "Heard reports of a major flood in {}. People need help.",
        "Water has entered our homes in {}.",
        "The river is overflowing its banks in {} area.",
    ],
    "TSUNAMI": [
        "The sea is receding unusually at {}. Is this a tsunami?",
        "Massive waves hitting the coast at {}. This looks like a tsunami warning.",
        "Official Tsunami alert for {} coast. Evacuate immediately!",
    ],
    "EARTHQUAKE": [
        "The ground is shaking violently in {}. Possible earthquake.",
        "My building in {} is swaying. I think it's an earthquake!",
        "Just felt a strong tremor in {}. Everything is rattling.",
    ],
    "LANDSLIDE": [
        "A huge landslide has blocked the road to {}.",
        "Mudslide reported in the hills above {}. Very dangerous.",
        "The road to {} is gone, taken by a landslide.",
    ],
    "CYCLONE": [
        "Cyclone warning for {}. Heavy winds and rain expected.",
        "The wind is howling in {}. Looks like the cyclone is here.",
        "Trees are falling down in {} due to the cyclonic storm.",
    ],
    "HIGH_WAVES": ["Giant waves are crashing on the shore at {}."],
    "STORM_SURGE": ["Sea water is rushing into the streets of {} due to the storm surge."],
    "FLASH_FLOOD": ["A sudden wall of water came down the valley near {}."]
}

class Command(BaseCommand):
    help = 'Seeds the database with realistic, clustered raw reports.'

    def add_arguments(self, parser):
        parser.add_argument('--number', type=int, help='The number of reports to create', default=10)

    def handle(self, *args, **options):
        fake = Faker('en_IN')
        number_of_reports = options['number']

        users = list(User.objects.all())
        if not users:
            self.stdout.write(self.style.ERROR('No users found. Please run `seed_users500` first.'))
            return

        all_hotspots = HOTSPOTS["coastal"] + HOTSPOTS["himalayan"] + HOTSPOTS["inland"]
        sources = ['Web', 'Mobile App', 'SMS', 'API']
        
        self.stdout.write(f'Creating {number_of_reports} new raw reports...')

        reports_to_create = []
        for _ in range(number_of_reports):
            # 75% of reports will be clustered around hotspots, 25% will be random noise
            if random.random() < 0.75:
                # --- Create a clustered report ---
                hotspot_name, (lat, lon), hazard_types = random.choice(all_hotspots)
                hazard_type = random.choice(hazard_types)
                
                # Generate a point near the hotspot with a small random offset
                latitude = random.gauss(lat, 0.15) # std deviation of ~15km
                longitude = random.gauss(lon, 0.15)
                location = Point(longitude, latitude, srid=4326)
                
                # Use a relevant report template
                template = random.choice(REPORT_TEMPLATES.get(hazard_type, REPORT_TEMPLATES["FLOOD"]))
                report_text = template.format(hotspot_name)

            else:
                # --- Create a random "noise" report ---
                lat_min, lat_max = 8.0, 37.0
                lon_min, lon_max = 68.0, 97.0
                latitude = random.uniform(lat_min, lat_max)
                longitude = random.uniform(lon_min, lon_max)
                location = Point(longitude, latitude, srid=4326)
                report_text = f"General alert near {fake.city()}. {fake.sentence()}"

            report = Reported(
                user=random.choice(users),
                report=report_text,
                is_marked=False, # Raw reports are initially not marked
                location=location,
                source=random.choice(sources)
            )
            reports_to_create.append(report)

        Reported.objects.bulk_create(reports_to_create)

        self.stdout.write(self.style.SUCCESS(f'Successfully created {number_of_reports} realistic raw reports.'))