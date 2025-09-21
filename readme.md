# Samudra Sathi: AI-Powered Geospatial Reporting API

[![Python Version][python-shield]][python-url]
[![Django Version][django-shield]][django-url]
[![License: MIT][license-shield]][license-url]

Samudra Sathi is a robust backend API designed for a real-time, AI-powered disaster and hazard reporting platform. It leverages the power of Django, Django Rest Framework, and PostGIS to provide a scalable, secure, and geospatially-aware service for collecting, processing, and visualizing incident reports.

## ✨ Features

-   🔐 **JWT Authentication:** Secure, token-based authentication (login, refresh, logout) using `djangorestframework-simplejwt`.
-   📝 **User & Incident Reporting:** Full CRUD API endpoints for managing users and their submitted incident reports.
-   🧠 **AI-Ready Architecture:** A clear data model distinction between raw user reports (`Reported`) and structured, AI-processed data (`CleanedReport`).
-   🗺️ **Advanced Geospatial API:**
    -   Serve location data in **GeoJSON** format, ready for modern map libraries (Mapbox, Leaflet).
    -   Filter endpoints by a geographic **bounding box** (`in_bbox`).
    -   High-performance **heatmap/density endpoints** that aggregate data on the fly using raw PostGIS queries.
-   🔧 **GIS-Enabled Admin Panel:** A user-friendly Django admin interface with OpenStreetMap widgets for easy location data management.
-   🚀 **Database Seeding:** Management commands to quickly populate the database with realistic test data for users and reports.

## 🛠️ Tech Stack

-   **Backend:** Django, Django Rest Framework
-   **Geospatial:** GeoDjango, PostGIS (PostgreSQL extension)
-   **Authentication:** DRF Simple JWT
-   **Database:** PostgreSQL
-   **Environment:** Python-dotenv

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need the following installed on your system:
*   Python 3.11+ and Pip
*   Git
*   PostgreSQL with PostGIS enabled.
    *   [Install PostgreSQL](https://www.postgresql.org/download/)
    *   [Enable PostGIS](https://postgis.net/install/) for your database (`CREATE EXTENSION postgis;`).
*   **GDAL Geospatial Library:** This is a critical dependency for GeoDjango.
    *   **Ubuntu/Debian:** `sudo apt-get install gdal-bin libgdal-dev`
    *   **macOS (with Homebrew):** `brew install gdal`
    *   **Windows:** This can be complex. The recommended way is to use the [OSGeo4W installer](https://trac.osgeo.org/osgeo4w/). The `settings.py` file in this project contains paths that assume this installation method.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Shoury-Rana/samudra-sathi.git
    cd samudra-sathi
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install `GDAL`:**
    ```
    Install wheel from https://github.com/cgohlke/geospatial-wheels/releases

    Should be similar to: gdal-<gdal_version>-cp<py_version>-cp<py_version>-win_amd64.whl)
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up environment variables:**
    Create a `.env` file in the project root. Copy the contents of the example below and fill in your database credentials.
    
    **.env.example**
    ```env
    SECRET_KEY='your-strong-secret-key-here'
    DEBUG=True
    ALLOWED_HOSTS=127.0.0.1,localhost

    # Database Credentials
    DB_NAME=samudra_db
    DB_USER=samudra_user
    DB_PASSWORD=samudra_pass
    DB_HOST=localhost
    DB_PORT=5432
    
    # CORS Origins
    CORS_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
    ```

6.  **Run database migrations:**
    ```bash
    python manage.py migrate
    ```

7.  **Create a superuser for the admin panel:**
    ```bash
    python manage.py createsuperuser
    ```

8.  **(Optional) Seed the database with sample data:**
    The project includes commands to populate the database with test data, which is highly recommended for exploring the API.
    ```bash
    # Create 500 users with random locations in India
    python manage.py seed_users500

    # Create 500 raw incident reports from those users
    python manage.py seed_reported

    # Create 500 pairs of raw and cleaned reports
    python manage.py seed_cleaned_reports
    ```

9.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```
    The API will be available at `http://127.0.0.1:8000/`.

---

##  API Endpoints Overview

Here is a summary of the main API endpoints available. For full request/response details, please refer to the complete API documentation.

| Endpoint                                  | Method       | Description                                          | Auth Required |
| ----------------------------------------- |--------------| ---------------------------------------------------- | :-----------: |
| `/api/users/register/`                    | `POST`       | Register a new user.                                 |       No      |
| `/api/users/login/`                       | `POST`       | Obtain JWT access and refresh tokens.                |       No      |
| `/api/users/logout/`                      | `POST`       | Log out by blacklisting the refresh token.           |      Yes      |
| `/api/reports/incidents/`                 | `GET`, `POST`| List your reports or create a new one.               |      Yes      |
| `/api/reports/incidents/{id}/`            | `GET`, `PUT`, `DELETE`| Manage a specific report you own.                  |      Yes      |
| `/api/reports/cleaned-incidents/`         | `GET`*, `POST`*| List or create AI-processed cleaned reports.         |      Yes      |
| `/api/reports/incidents/locations/`       | `GET`        | Get raw report locations as GeoJSON.                 |      Yes      |
| `/api/reports/incidents/density/`         | `GET`        | Get heatmap density data for raw reports.            |      Yes      |
| `/api/reports/cleaned-incidents/locations/`| `GET`        | Get cleaned report locations as GeoJSON.             |      Yes      |
| `/api/reports/cleaned-incidents/density/` | `GET`        | Get heatmap density data for cleaned reports.        |      Yes      |


## 📂 Project Structure

The project is organized into two main Django apps:

-   `users/`: Handles all user-related logic, including the custom user model, authentication, and user location endpoints.
-   `reports/`: Manages all incident reporting logic, including the `Reported` and `CleanedReport` models, serializers, and all geospatial API views.


[python-shield]: https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[django-shield]: https://img.shields.io/badge/Django-5.x-092E20?style=for-the-badge&logo=django&logoColor=white
[django-url]: https://www.djangoproject.com/
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template?style=for-the-badge
[license-url]: https://github.com/your-username/samudra-sathi/blob/main/LICENSE