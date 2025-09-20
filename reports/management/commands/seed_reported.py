import random
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from faker import Faker
from users.models import User
from reports.models import Reported

class Command(BaseCommand):
    help = 'Seeds the database with random raw reports'

    def add_arguments(self, parser):
        parser.add_argument('--number', type=int, help='The number of reports to create', default=500)

    def handle(self, *args, **options):
        fake = Faker('en_IN')
        number_of_reports = options['number']

        # Fetch all users to avoid repeated DB queries inside the loop
        users = list(User.objects.all())
        if not users:
            self.stdout.write(self.style.ERROR('No users found in the database. Please seed users first.'))
            return

        # Bounding box for India (approximate)
        lat_min, lat_max = 8.0, 37.0
        lon_min, lon_max = 68.0, 97.0

        # Sample report templates
        report_templates = [
            "Seeing high water levels near {}. It's rising fast!",
            "Massive waves hitting the coast at {}. This looks like a tsunami warning.",
            "The ground is shaking violently in {}. Possible earthquake.",
            "Heard reports of a major flood in {}. People need help.",
            "My building in {} is swaying. I think it's an earthquake!",
            "Water has entered our homes in {}.",
            "The sea is receding unusually at {}. Is this a tsunami?",
        ]
        
        sources = ['Web', 'Mobile App', 'SMS', 'API']
        
        self.stdout.write(f'Creating {number_of_reports} new raw reports...')

        reports_to_create = []
        for _ in range(number_of_reports):
            # Generate random location
            latitude = random.uniform(lat_min, lat_max)
            longitude = random.uniform(lon_min, lon_max)
            location = Point(longitude, latitude, srid=4326)

            report_text = random.choice(report_templates).format(fake.city())

            report = Reported(
                user=random.choice(users),
                report=report_text,
                is_marked=False, # Raw reports are initially not marked
                location=location,
                source=random.choice(sources)
            )
            reports_to_create.append(report)

        Reported.objects.bulk_create(reports_to_create)

        self.stdout.write(self.style.SUCCESS(f'Successfully created {number_of_reports} raw reports.'))