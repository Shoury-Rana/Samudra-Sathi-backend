import random
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from faker import Faker
from users.models import User

class Command(BaseCommand):
    help = 'Seeds the database with random users'

    def add_arguments(self, parser):
        parser.add_argument('--number', type=int, help='The number of users to create', default=15)

    def handle(self, *args, **options):
        fake = Faker('en_IN') # Use Indian locale for names
        number_of_users = options['number']
        
        # Bounding box for India (approximate)
        lat_min, lat_max = 8.0, 37.0
        lon_min, lon_max = 68.0, 97.0

        self.stdout.write(f'Creating {number_of_users} new users...')

        for _ in range(number_of_users):
            profile = fake.profile()
            first_name, last_name = profile['name'].split(' ', 1)
            
            # Generate a unique email
            email = f"user.{fake.unique.user_name()}@example.com"

            # Generate random location
            latitude = random.uniform(lat_min, lat_max)
            longitude = random.uniform(lon_min, lon_max)
            
            user = User.objects.create_user(
                email=email,
                password='password123', # Use a standard password for all test users
                first_name=first_name,
                last_name=last_name,
                location=Point(longitude, latitude, srid=4326)
            )

        self.stdout.write(self.style.SUCCESS(f'Successfully created {number_of_users} users.'))
