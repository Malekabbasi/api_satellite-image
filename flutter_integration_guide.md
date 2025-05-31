# Guide d'Intégration Flutter - API Satellite Améliorée

## 🔄 Mise à Jour Effectuée

### URL API Mise à Jour
- **Ancienne URL** : `https://malek1ab-ndvi-api.hf.space`
- **Nouvelle URL** : `https://user:d26b803a1a779d115247b92951085486@satellite-image-api-tunnel-817mn8j3.devinapps.com`

### ✅ Fonctions Préservées
Tous les noms de fonctions globales sont conservés :
- `_checkApiHealth()`
- `_fetchAvailableMethods()`
- `_callCalculateIndicesApi()`
- `_callCalculateTimeSeriesApi()`
- `_pickGeoJsonFile()`
- `_launchAnalysis()`

### 🆕 Nouveaux Endpoints Disponibles
- `/calculate-indices-geojson` - Retourne GeoJSON avec masques
- `/calculate-masks-geojson` - Masques de végétation colorés
- Support automatique avec fallback vers anciens endpoints

### 📊 Indices Supportés
- ✅ **NDVI** - Santé végétation générale
- ✅ **NDWI** - Contenu en eau
- ✅ **SAVI** - Végétation ajustée pour le sol
- ✅ **EVI** - Végétation améliorée

## 🚀 Instructions d'Utilisation

1. **Remplacer le fichier** : Utilisez le fichier `satellite_monitoring_page.dart` mis à jour
2. **Aucune modification** : Tous les noms de fonctions restent identiques
3. **Test automatique** : L'API utilise automatiquement les nouveaux endpoints optimisés
4. **Compatibilité** : Fallback automatique vers anciens endpoints si nécessaire

## 🔗 Lien API Public
```
https://user:d26b803a1a779d115247b92951085486@satellite-image-api-tunnel-817mn8j3.devinapps.com
```

Le code Flutter est maintenant prêt à utiliser l'API satellite améliorée avec toutes les nouvelles fonctionnalités !
