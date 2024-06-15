import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img;
// import 'package:tflite/tflite.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras[1];

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
  Interpreter? model;
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
    model = await Interpreter.fromAsset(
      "assets/antispoofing_model.tflite",
    );
    model?.allocateTensors();
  }

  @override
  void dispose() {
    _controller.dispose();
    faceDetector.close();
    model?.close();
    super.dispose();
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
            _controller.startImageStream((CameraImage image) async {
              if (!canCapture) return;
              canCapture = false;
              final XFile imageXFile = await _controller.takePicture();
              final faces = await detectFaces(imageXFile);
              await runModelOnFrame(image, faces);
              canCapture = true;
            });
          } catch (e) {
            print(e);
          }
        },
        child: const Icon(Icons.camera),
      ),
    );
  }

  bool canCapture = true;

  void drawBoundingBox(int left, int top, int width, int height, Color color) {
    setState(() {
      rects.add(Rect.fromLTWH(left.toDouble(), top.toDouble(), width.toDouble(),
          height.toDouble()));
      colors.add(color);
    });
  }

  Future<void> runModelOnFrame(CameraImage image, List<Face> faces) async {
    for (Face face in faces) {
      final left = face.boundingBox.left.toInt();
      final top = face.boundingBox.top.toInt();
      final width = face.boundingBox.width.toInt();
      final height = face.boundingBox.height.toInt();
      List<double> result = [];
      try {
        // var imgBytes = concatenatePlanes(img.planes).buffer.asUint8List();

        var imgBytes = convertCameraImageToUint8List(image);
        model?.run(
          imgBytes,
          result,
        );
      } catch (e) {
        canCapture = true;
        print("Mohamed: ${e.toString()}");
      }
      print("Mohamed: ${result.first}");
      if (result.isNotEmpty) {
        setState(() {
          if (result[0] > 0.5) {
            drawBoundingBox(left, top, width, height, Colors.red);
          } else {
            drawBoundingBox(left, top, width, height, Colors.green);
          }
        });
      }
    }
  }

  Uint8List concatenatePlanes(List<Plane> planes) {
    WriteBuffer allBytes = WriteBuffer();
    for (Plane plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  Future<List<Face>> detectFaces(XFile imageFile) async {
    final InputImage inputImage = InputImage.fromFilePath(imageFile.path);
    final faces = await faceDetector.processImage(inputImage);
    return faces;
  }

  Uint8List convertCameraImageToUint8List(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    final img.Image rgbImage = img.Image(width: width, height: height);

    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * width + x;

        final int yValue = image.planes[0].bytes[index];
        final int uValue = image.planes[1].bytes[uvIndex];
        final int vValue = image.planes[2].bytes[uvIndex];

        int r = (yValue + (1.370705 * (vValue - 128))).round();
        int g =
            (yValue - (0.337633 * (uValue - 128)) - (0.698001 * (vValue - 128)))
                .round();
        int b = (yValue + (1.732446 * (uValue - 128))).round();

        rgbImage.setPixel(x, y, img.ColorFloat16.rgb(r, g, b));
      }
    }

    return Uint8List.fromList(img.encodeJpg(rgbImage));
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
