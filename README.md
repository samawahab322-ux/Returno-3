# Returno - Missing Persons Platform

Returno is a Flask-based web application designed to help find missing persons. It uses AI-powered face recognition to match photos of missing individuals with a database of reports.

## Features

*   **Report Missing Persons:** Submit detailed reports for missing individuals, including personal information, descriptions, and photos.
*   **AI-Powered Search:** Search for a missing person by uploading a photo. The application uses a machine learning model to predict the identity.
*   **Database Storage:** All reports are stored in a persistent SQLite database.
*   **Dashboard:** A simple dashboard to view all submitted reports.
*   **Live Recognition:** A live video stream for real-time face recognition (requires a webcam).

## Prerequisites

*   Python 3.x
*   pip (Python package installer)

## Installation

1.  **Clone the repository** (or download the source code).

2.  **Navigate to the project directory:**
    ```bash
    cd returno
    ```

3.  **Install the required Python packages** by running the following command:
    ```bash
    pip install -r requirements.txt
    ```

## Database Setup

The application uses a SQLite database. The database file, named `database.db`, will be automatically created in the project directory the first time you run the application. No manual setup is required.

## Running the Application

1.  **Run the Flask server** with the following command:
    ```bash
    python app.py
    ```

2.  **Access the application** by opening your web browser and navigating to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Application Endpoints

*   **`GET /`**: The main landing page where you can navigate to other sections.
*   **`GET /dashboard`**: Displays a table with all the missing person reports stored in the database.
*   **`POST /api/report_missing`**: The API endpoint for submitting a new missing person report. The frontend form sends the data here.
*   **`POST /api/search_by_photo`**: API endpoint to search for a person by uploading a photo.
*   **`GET /api/live_recognition`**: Provides a live video stream from a webcam with real-time face recognition.
*   **`GET /api/get_missing_reports`**: Returns a JSON response containing all missing person reports.
*   **`GET /api/search_by_name`**: Searches for reports by the missing person's name via a query parameter (e.g., `/api/search_by_name?name=John`).
