import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';
import 'package:http_parser/http_parser.dart';
import 'package:intl/intl.dart';

enum AnalysisType { single, timeSeries }

class SatelliteMonitoringPage extends StatefulWidget {
  const SatelliteMonitoringPage({super.key});

  @override
  State<SatelliteMonitoringPage> createState() =>
      _SatelliteMonitoringPageState();
}

class _SatelliteMonitoringPageState extends State<SatelliteMonitoringPage> {
  // Configuration API - Updated to new enhanced API
  static const String apiUrl = 'https://user:d26b803a1a779d115247b92951085486@satellite-image-api-tunnel-817mn8j3.devinapps.com';

  // États
  bool _isLoading = false;
  String? _status;
  Map<String, dynamic>? _result;
  String? _error;
  String _loadingMessage = '';
  AnalysisType _selectedAnalysis = AnalysisType.single;

  // Variables pour l'upload de fichier
  Uint8List? _selectedFileBytes;
  String? _selectedFileName;
  bool _useUploadedFile = false;

  // Variables pour les séries temporelles
  DateTime _startDate = DateTime.now().subtract(const Duration(days: 90));
  DateTime _endDate = DateTime.now();
  final TextEditingController _intervalController =
      TextEditingController(text: '15');
  List<String> _selectedIndices = ['ndvi']; // NDVI par défaut
  final List<String> _availableIndices = ['ndvi', 'ndwi', 'savi', 'evi'];

  // Méthode d'amélioration sélectionnée
  String _selectedEnhancementMethod = 'adaptive';
  Map<String, dynamic>? _availableMethods;

  // GeoJSON de test - Zone Tunis
  final String _testGeoJson = '''
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"name": "Zone test Tunis"},
    "geometry": {
      "coordinates": [[[9.489834030582017,35.34669303400706],[9.49068121621525,35.34622919916441],[9.49142395406065,35.34584582159722],[9.492352376366767,35.34702907983558],[9.490814676921133,35.34786681618888],[9.489834030582017,35.34669303400706]]],
      "type": "Polygon"
    }
  }]
}''';

  @override
  void initState() {
    super.initState();
    _checkApiHealth();
    _fetchAvailableMethods();
  }

  @override
  void dispose() {
    _intervalController.dispose();
    super.dispose();
  }

  Future<void> _checkApiHealth() async {
    setState(() => _status = 'Vérification en cours...');
    try {
      final response = await http
          .get(Uri.parse('$apiUrl/health'))
          .timeout(const Duration(seconds: 15));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _status = '✅ ${data['status']} - v${data['version']}';
        });
      } else {
        setState(() {
          _status = '❌ Erreur ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _status = '❌ Erreur de connexion: $e';
      });
    }
  }

  Future<void> _fetchAvailableMethods() async {
    try {
      final response = await http
          .get(Uri.parse('$apiUrl/enhancement-methods'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        setState(() {
          _availableMethods = jsonDecode(response.body);
        });
      }
    } catch (e) {
      debugPrint('Erreur lors de la récupération des méthodes: $e');
    }
  }

  Future<void> _pickGeoJsonFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['geojson', 'json'],
        withData: true,
      );

      if (result != null && result.files.single.bytes != null) {
        setState(() {
          _selectedFileBytes = result.files.single.bytes;
          _selectedFileName = result.files.single.name;
          _useUploadedFile = true;
        });
        _showSuccess('✅ Fichier GeoJSON sélectionné: ${result.files.single.name}');
      }
    } catch (e) {
      _showError('Erreur lors de la sélection du fichier: $e');
    }
  }

  // Enhanced function with GeoJSON endpoint support while preserving original name
  Future<void> _callCalculateIndicesApi() async {
    try {
      // Try new GeoJSON endpoint first for enhanced features
      await _callCalculateIndicesGeoJsonApi();
    } catch (e) {
      debugPrint('GeoJSON endpoint failed, using standard endpoint: $e');
      // Fallback to original endpoint for compatibility
      await _callCalculateIndicesStandardApi();
    }
  }

  // New GeoJSON endpoint function
  Future<void> _callCalculateIndicesGeoJsonApi() async {
    final uri = Uri.parse('$apiUrl/calculate-indices-geojson');
    final request = http.MultipartRequest('POST', uri);

    // Add enhancement method as form field
    request.fields['enhancement_method'] = _selectedEnhancementMethod;
    
    // Add indices as form fields
    for (String index in _availableIndices) {
      request.fields['indices'] = index;
    }

    // Add GeoJSON file
    if (_useUploadedFile && _selectedFileBytes != null) {
      request.files.add(
        http.MultipartFile.fromBytes(
          'geojson',
          _selectedFileBytes!,
          filename: _selectedFileName ?? 'uploaded.geojson',
          contentType: MediaType('application', 'json'),
        ),
      );
    } else {
      request.files.add(
        http.MultipartFile.fromBytes(
          'geojson',
          utf8.encode(_testGeoJson),
          filename: 'test_tunis.geojson',
          contentType: MediaType('application', 'json'),
        ),
      );
    }

    final streamedResponse =
        await request.send().timeout(const Duration(minutes: 5));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final result = jsonDecode(utf8.decode(response.bodyBytes));
      setState(() {
        _result = result;
        _isLoading = false;
      });
      _showSuccess('🎉 Analyse d\'image terminée avec nouvelles fonctionnalités !');
    } else {
      throw Exception('GeoJSON endpoint error: ${response.statusCode} - ${response.body}');
    }
  }

  // Original endpoint function for fallback compatibility
  Future<void> _callCalculateIndicesStandardApi() async {
    final uri = Uri.parse(
        '$apiUrl/calculate-indices?enhancement_method=$_selectedEnhancementMethod');
    final request = http.MultipartRequest('POST', uri);

    // Ajouter le GeoJSON (personnel ou de test)
    if (_useUploadedFile && _selectedFileBytes != null) {
      request.files.add(
        http.MultipartFile.fromBytes(
          'geojson',
          _selectedFileBytes!,
          filename: _selectedFileName ?? 'uploaded.geojson',
          contentType: MediaType('application', 'json'),
        ),
      );
    } else {
      request.files.add(
        http.MultipartFile.fromBytes(
          'geojson',
          utf8.encode(_testGeoJson),
          filename: 'test_tunis.geojson',
          contentType: MediaType('application', 'json'),
        ),
      );
    }

    try {
      final streamedResponse =
          await request.send().timeout(const Duration(minutes: 5));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final result = jsonDecode(utf8.decode(response.bodyBytes));
        setState(() {
          _result = result;
          _isLoading = false;
        });
        _showSuccess('🎉 Analyse d\'image terminée !');
      } else {
        setState(() {
          _error = 'Erreur ${response.statusCode}: ${response.body}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error =
            'Erreur lors de l\'appel API: $e. Le traitement peut être long, réessayez.';
        _isLoading = false;
      });
    }
  }

  Future<void> _callCalculateTimeSeriesApi() async {
    // Construire l'URL avec les paramètres de requête
    final queryParams = {
      'start_date': DateFormat('yyyy-MM-dd').format(_startDate),
      'end_date': DateFormat('yyyy-MM-dd').format(_endDate),
      'interval_days': _intervalController.text,
      'enhancement_method': _selectedEnhancementMethod,
      'create_animation': 'true',
      // Ajouter chaque indice séparément
      ..._selectedIndices.asMap().map((_, value) => MapEntry('indices', value)),
    };

    final uri = Uri.parse('$apiUrl/calculate-time-series')
        .replace(queryParameters: queryParams);
    final request = http.MultipartRequest('POST', uri);

    // Ajouter le GeoJSON (personnel ou de test)
    if (_useUploadedFile && _selectedFileBytes != null) {
      request.files.add(
        http.MultipartFile.fromBytes(
          'geojson',
          _selectedFileBytes!,
          filename: _selectedFileName ?? 'uploaded.geojson',
          contentType: MediaType('application', 'json'),
        ),
      );
    } else {
      request.files.add(
        http.MultipartFile.fromBytes(
          'geojson',
          utf8.encode(_testGeoJson),
          filename: 'test_tunis.geojson',
          contentType: MediaType('application', 'json'),
        ),
      );
    }

    try {
      final streamedResponse =
          await request.send().timeout(const Duration(minutes: 10));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final result = jsonDecode(utf8.decode(response.bodyBytes));
        setState(() {
          _result = result;
          _isLoading = false;
        });
        _showSuccess('🎉 Analyse de série temporelle terminée !');
      } else {
        setState(() {
          _error = 'Erreur ${response.statusCode}: ${response.body}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error =
            'Erreur lors de l\'appel API: $e. Le traitement peut être long, réessayez.';
        _isLoading = false;
      });
    }
  }

  Future<void> _launchAnalysis() async {
    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
      _loadingMessage = _selectedAnalysis == AnalysisType.single
          ? 'Analyse d\'image en cours...'
          : 'Analyse de série temporelle en cours...';
    });

    try {
      if (_selectedAnalysis == AnalysisType.single) {
        await _callCalculateIndicesApi();
      } else {
        await _callCalculateTimeSeriesApi();
      }
    } catch (e) {
      setState(() {
        _error = 'Erreur lors de l\'analyse: $e';
        _isLoading = false;
      });
    }
  }

  void _showSuccess(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 5),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Surveillance Satellite'),
        backgroundColor: Colors.green[700],
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status de l'API
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'État de l\'API',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    Text(_status ?? 'Vérification...'),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Sélection du type d'analyse
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Type d\'analyse',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    SegmentedButton<AnalysisType>(
                      segments: const [
                        ButtonSegment<AnalysisType>(
                          value: AnalysisType.single,
                          label: Text('Image unique'),
                          icon: Icon(Icons.image),
                        ),
                        ButtonSegment<AnalysisType>(
                          value: AnalysisType.timeSeries,
                          label: Text('Série temporelle'),
                          icon: Icon(Icons.timeline),
                        ),
                      ],
                      selected: {_selectedAnalysis},
                      onSelectionChanged: (Set<AnalysisType> newSelection) {
                        setState(() {
                          _selectedAnalysis = newSelection.first;
                        });
                      },
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Configuration GeoJSON
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Zone d\'analyse (GeoJSON)',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            _useUploadedFile && _selectedFileName != null
                                ? 'Fichier: $_selectedFileName'
                                : 'Zone de test (Tunis)',
                            style: const TextStyle(fontSize: 14),
                          ),
                        ),
                        ElevatedButton.icon(
                          onPressed: _pickGeoJsonFile,
                          icon: const Icon(Icons.upload_file),
                          label: const Text('Choisir fichier'),
                        ),
                      ],
                    ),
                    if (_useUploadedFile)
                      TextButton(
                        onPressed: () {
                          setState(() {
                            _useUploadedFile = false;
                            _selectedFileBytes = null;
                            _selectedFileName = null;
                          });
                        },
                        child: const Text('Utiliser zone de test'),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Configuration pour série temporelle
            if (_selectedAnalysis == AnalysisType.timeSeries) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Configuration série temporelle',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 16),
                      
                      // Dates
                      Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text('Date de début'),
                                const SizedBox(height: 4),
                                InkWell(
                                  onTap: () async {
                                    final date = await showDatePicker(
                                      context: context,
                                      initialDate: _startDate,
                                      firstDate: DateTime(2015),
                                      lastDate: DateTime.now(),
                                    );
                                    if (date != null) {
                                      setState(() => _startDate = date);
                                    }
                                  },
                                  child: Container(
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      border: Border.all(color: Colors.grey),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(DateFormat('yyyy-MM-dd').format(_startDate)),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text('Date de fin'),
                                const SizedBox(height: 4),
                                InkWell(
                                  onTap: () async {
                                    final date = await showDatePicker(
                                      context: context,
                                      initialDate: _endDate,
                                      firstDate: _startDate,
                                      lastDate: DateTime.now(),
                                    );
                                    if (date != null) {
                                      setState(() => _endDate = date);
                                    }
                                  },
                                  child: Container(
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      border: Border.all(color: Colors.grey),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(DateFormat('yyyy-MM-dd').format(_endDate)),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      
                      // Intervalle
                      TextField(
                        controller: _intervalController,
                        decoration: const InputDecoration(
                          labelText: 'Intervalle (jours)',
                          border: OutlineInputBorder(),
                          helperText: 'Nombre de jours entre chaque image',
                        ),
                        keyboardType: TextInputType.number,
                      ),
                      const SizedBox(height: 16),
                      
                      // Sélection des indices
                      const Text('Indices de végétation'),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 8,
                        children: _availableIndices.map((index) {
                          return FilterChip(
                            label: Text(index.toUpperCase()),
                            selected: _selectedIndices.contains(index),
                            onSelected: (selected) {
                              setState(() {
                                if (selected) {
                                  _selectedIndices.add(index);
                                } else {
                                  _selectedIndices.remove(index);
                                }
                              });
                            },
                          );
                        }).toList(),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Méthode d'amélioration
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Méthode d\'amélioration',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    if (_availableMethods != null) ...[
                      DropdownButtonFormField<String>(
                        value: _selectedEnhancementMethod,
                        decoration: const InputDecoration(
                          border: OutlineInputBorder(),
                        ),
                        items: (_availableMethods!['methods'] as List<dynamic>)
                            .map<DropdownMenuItem<String>>((method) {
                          return DropdownMenuItem<String>(
                            value: method['name'],
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(method['name']),
                                Text(
                                  method['description'],
                                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                                ),
                              ],
                            ),
                          );
                        }).toList(),
                        onChanged: (String? newValue) {
                          if (newValue != null) {
                            setState(() {
                              _selectedEnhancementMethod = newValue;
                            });
                          }
                        },
                      ),
                    ] else ...[
                      const CircularProgressIndicator(),
                      const SizedBox(height: 8),
                      const Text('Chargement des méthodes...'),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Bouton de lancement
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _launchAnalysis,
              icon: _isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.play_arrow),
              label: Text(_isLoading ? _loadingMessage : 'Lancer l\'analyse'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green[700],
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                textStyle: const TextStyle(fontSize: 16),
              ),
            ),
            const SizedBox(height: 24),

            // Affichage des erreurs
            if (_error != null) ...[
              Card(
                color: Colors.red[50],
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(Icons.error, color: Colors.red),
                          SizedBox(width: 8),
                          Text(
                            'Erreur',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.red,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(_error!),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],

            // Affichage des résultats
            if (_result != null) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Résultats',
                        style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 16),
                      
                      // Affichage des images si disponibles
                      if (_result!['images'] != null) ...[
                        const Text(
                          'Images générées:',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                        ),
                        const SizedBox(height: 8),
                        ...(_result!['images'] as Map<String, dynamic>).entries.map((entry) {
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 16.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  entry.key,
                                  style: const TextStyle(fontWeight: FontWeight.w500),
                                ),
                                const SizedBox(height: 4),
                                if (entry.value is String && entry.value.startsWith('data:image'))
                                  Image.memory(
                                    base64Decode(entry.value.split(',')[1]),
                                    height: 200,
                                    fit: BoxFit.contain,
                                  )
                                else
                                  Text('Image: ${entry.value}'),
                              ],
                            ),
                          );
                        }).toList(),
                      ],
                      
                      // Affichage des statistiques si disponibles
                      if (_result!['statistics'] != null) ...[
                        const SizedBox(height: 16),
                        const Text(
                          'Statistiques:',
                          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                        ),
                        const SizedBox(height: 8),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.grey[100],
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            const JsonEncoder.withIndent('  ').convert(_result!['statistics']),
                            style: const TextStyle(fontFamily: 'monospace'),
                          ),
                        ),
                      ],
                      
                      // Affichage des données brutes
                      const SizedBox(height: 16),
                      ExpansionTile(
                        title: const Text('Données complètes (JSON)'),
                        children: [
                          Container(
                            width: double.infinity,
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.grey[100],
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: SingleChildScrollView(
                              scrollDirection: Axis.horizontal,
                              child: Text(
                                const JsonEncoder.withIndent('  ').convert(_result!),
                                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
