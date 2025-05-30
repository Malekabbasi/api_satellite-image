---
title: Satellite Vegetation Indices API
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false

# API Satellite Image Processing - Version Améliorée 3.0.0

Cette API permet de traiter des images satellites pour calculer des indices de végétation avec support Flutter et optimisations avancées.

## 🚀 Nouvelles Fonctionnalités

### Support Flutter Intégré
- **Endpoints GeoJSON** : `/calculate-indices-geojson` et `/calculate-masks-geojson`
- **Système de coordonnées WGS84** pour compatibilité avec les cartes Flutter
- **Propriétés de style** pour le rendu des cartes
- **Classification automatique** de la végétation avec couleurs

### Optimisations de Performance
- **Calculs vectorisés** avec NumPy pour les indices de végétation
- **Cache Redis** pour les requêtes SentinelHub (optionnel)
- **Traitement asynchrone** pour les séries temporelles
- **Gestion mémoire optimisée** pour les grandes images

### Sécurité Renforcée
- **Validation Pydantic** pour toutes les entrées
- **Limitation de débit** (rate limiting) avec SlowAPI
- **Validation de fichiers** avec vérification MIME
- **IDs de corrélation** pour le suivi des requêtes

## 📋 Indices de Végétation Supportés

| Indice | Description | Plage de Valeurs |
|--------|-------------|------------------|
| **NDVI** | Normalized Difference Vegetation Index | -1.0 à 1.0 |
| **NDWI** | Normalized Difference Water Index | -1.0 à 1.0 |
| **SAVI** | Soil Adjusted Vegetation Index | -1.5 à 1.5 |
| **EVI** | Enhanced Vegetation Index | -1.0 à 1.0 |

## 🗺️ Endpoints API

### Endpoints Classiques
- `GET /` - Informations sur l'API
- `GET /health` - Vérification de l'état
- `POST /calculate-indices` - Calcul des indices (format classique)

### Nouveaux Endpoints GeoJSON (Flutter)
- `POST /calculate-indices-geojson` - Points GeoJSON avec valeurs d'indices
- `POST /calculate-masks-geojson` - Polygones GeoJSON pour masques de végétation

### Endpoints Utilitaires
- `GET /enhancement-methods` - Méthodes d'amélioration disponibles
- `GET /health/detailed` - Vérification détaillée du système

## 🛠️ Installation et Configuration

### Prérequis
```bash
pip install -r requirements.txt
```

### Variables d'Environnement
```bash
# Configuration SentinelHub (obligatoire)
SENTINELHUB_INSTANCE_ID=your_instance_id
SENTINELHUB_CLIENT_ID=your_client_id
SENTINELHUB_CLIENT_SECRET=your_client_secret

# Configuration Redis (optionnel)
REDIS_URL=redis://localhost:6379
```

### Démarrage
```bash
python app.py
```

## 📱 Intégration Flutter

### Exemple d'utilisation des endpoints GeoJSON
```dart
// Calcul des indices avec points GeoJSON
final response = await http.post(
  Uri.parse('$baseUrl/calculate-indices-geojson'),
  body: formData,
);

final geojsonData = jsonDecode(response.body);
// Utiliser geojsonData avec flutter_map
```

### Structure des réponses GeoJSON
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [lon, lat]},
      "properties": {
        "ndvi_value": 0.75,
        "ndvi_class": "dense_vegetation",
        "vegetation_level": "high_vegetation",
        "color": "#2e7d32"
      }
    }
  ],
  "metadata": {
    "coordinate_system": "WGS84",
    "feature_count": 25
  }
}
```

---
