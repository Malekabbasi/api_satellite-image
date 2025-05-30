#!/usr/bin/env python3
"""
Exemple de script pour tester l'API des indices de végétation
"""

import requests
import json
import base64
import os
from PIL import Image
from io import BytesIO

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test du point de contrôle de santé"""
    print("🔍 Test du point de contrôle de santé...")
    response = requests.get(f"{API_BASE_URL}/")
    
    if response.status_code == 200:
        print("✅ API en fonctionnement")
        print(f"   Réponse: {response.json()}")
    else:
        print(f"❌ Erreur: {response.status_code}")
    print()

def test_api_info():
    """Test du point d'information de l'API"""
    print("🔍 Test des informations de l'API...")
    response = requests.get(f"{API_BASE_URL}/info")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Informations récupérées avec succès")
        print(f"   Indices supportés: {list(data['indices_supportes'].keys())}")
        print(f"   Données satellite: {data['donnees_satellite']}")
    else:
        print(f"❌ Erreur: {response.status_code}")
    print()

def create_sample_geojson():
    """Crée un exemple de fichier GeoJSON pour les tests"""
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [10.0, 36.0],  # Tunisie - zone agricole
                        [10.1, 36.0],
                        [10.1, 36.1],
                        [10.0, 36.1],
                        [10.0, 36.0]
                    ]]
                },
                "properties": {
                    "name": "Zone de test Tunisie"
                }
            }
        ]
    }
    
    filename = "test_zone.geojson"
    with open(filename, 'w') as f: