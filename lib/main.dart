import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:tflite/tflite.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({Key? key, required this.camera}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark(),
      home: FaceDetectionScreen(camera: camera),
    );
  }
}

class FaceDetectionScreen extends StatefulWidget {
  final CameraDescription camera;

  const FaceDetectionScreen({Key? key, required this.camera}) : super(key: key);

  @override
  _FaceDetectionScreenState createState() => _FaceDetectionScreenState();
}

class _FaceDetectionScreenState extends State<FaceDetectionScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  // ignore: deprecated_member_use
  final FaceDetector faceDetector = GoogleMlKit.vision.faceDetector();
  bool isDetecting = false;
  List<Rect> rects = [];
  List<Color> colors = [];

  CameraImage? get cameraImage => null;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
    );
    _initializeControllerFuture = _controller.initialize();
    loadModel();
  }

  Future<void> loadModel() async {
    await Tflite.loadModel(
      model: "assets/antispoofing_model.tflite",
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    faceDetector.close();
    Tflite.close();
    super.dispose();
  }

  Future<void> runModelOnFrame(CameraImage img, List<Face> faces) async {
    for (Face face in faces) {
      final left = face.boundingBox.left.toInt();
      final top = face.boundingBox.top.toInt();
      final width = face.boundingBox.width.toInt();
      final height = face.boundingBox.height.toInt();

      var imgBytes = concatenatePlanes(img.planes);

      var result = await Tflite.runModelOnBinary(
        binary: imgBytes.buffer.asUint8List(),
        numResults: 1,
      );

      setState(() {
        if (result![0]["label"] == "spoof") {
          drawBoundingBox(left, top, width, height, Colors.red);
        } else {
          drawBoundingBox(left, top, width, height, Colors.green);
        }
      });
    }
  }

  Uint8List concatenatePlanes(List<Plane> planes) {
    WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  void drawBoundingBox(int left, int top, int width, int height, Color color) {
    setState(() {
      rects.add(Rect.fromLTWH(left.toDouble(), top.toDouble(), width.toDouble(),
          height.toDouble()));
      colors.add(color);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Detection')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              children: [
                CameraPreview(_controller),
                CustomPaint(
                  painter: BoundingBoxPainter(rects, colors),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          try {
            await _initializeControllerFuture;
            final image = await _controller.takePicture();
            final faces = await detectFaces(image);
            // Convert the image to CameraImage format before passing to runModelOnFrame
            // Note: You may need to implement a method to convert XFile to CameraImage
            await runModelOnFrame(cameraImage!, faces);
          } catch (e) {
            print(e);
          }
        },
        child: Icon(Icons.camera),
      ),
    );
  }

  Future<List<Face>> detectFaces(XFile imageFile) async {
    final InputImage inputImage = InputImage.fromFilePath(imageFile.path);
    final faces = await faceDetector.processImage(inputImage);
    return faces;
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<Rect> rects;
  final List<Color> colors;

  BoundingBoxPainter(this.rects, this.colors);

  @override
  void paint(Canvas canvas, Size size) {
    for (int i = 0; i < rects.length; i++) {
      final rect = rects[i];
      final color = colors[i];
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
