import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import os, json
from dotenv import load_dotenv, dotenv_values


class DbMain:
    def __init__(self):
        load_dotenv()
        self.serviceAccountKey = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.base_url = os.getenv("BASE_URL")
        self.cred = credentials.Certificate(self.serviceAccountKey)
        firebase_admin.initialize_app(self.cred, {
            'databaseURL': self.base_url
        })
        self.ref = db.reference("/violence_location")


    def send_lat_and_long(self, lat, long):
        data = {
            "lat":lat,
            "long":long
        }
        self.ref.push(data)
