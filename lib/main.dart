import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as img;
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

  void drawFocusCorners(Canvas canvas, Rect rect, Color color) {
    final double cornerLength = 20.0;
    final double thickness = 3.0;

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = thickness;

    final corners = [
      // Top-left corner
      [
        Offset(rect.left, rect.top),
        Offset(rect.left + cornerLength, rect.top),
        Offset(rect.left, rect.top + cornerLength),
      ],
      // Top-right corner
      [
        Offset(rect.right, rect.top),
        Offset(rect.right - cornerLength, rect.top),
        Offset(rect.right, rect.top + cornerLength),
      ],
      // Bottom-left corner
      [
        Offset(rect.left, rect.bottom),
        Offset(rect.left + cornerLength, rect.bottom),
        Offset(rect.left, rect.bottom - cornerLength),
      ],
      // Bottom-right corner
      [
        Offset(rect.right, rect.bottom),
        Offset(rect.right - cornerLength, rect.bottom),
        Offset(rect.right, rect.bottom - cornerLength),
      ],
    ];

    for (var corner in corners) {
      canvas.drawLine(corner[0], corner[1], paint);
      canvas.drawLine(corner[0], corner[2], paint);
    }
  }

  Future<void> runModelOnFrame(CameraImage image, List<Face> faces) async {
    for (Face face in faces) {
      final left = face.boundingBox.left.toInt();
      final top = face.boundingBox.top.toInt();
      final width = face.boundingBox.width.toInt();
      final height = face.boundingBox.height.toInt();
      List<double> result = List.filled(1, 0.0);
      try {
        var imgBytes = preprocessFace(image, left, top, width, height);
        model?.run(
          imgBytes,
          result,
        );
      } catch (e) {
        canCapture = true;
        print("Error: ${e.toString()}");
      }
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

  void drawBoundingBox(int left, int top, int width, int height, Color color) {
    setState(() {
      rects.add(Rect.fromLTWH(left.toDouble(), top.toDouble(), width.toDouble(),
          height.toDouble()));
      colors.add(color);
    });
  }

  img.Image copyCrop(img.Image src, int x, int y, int w, int h) {
    // Ensure cropping dimensions are within image bounds
    int startX = x < 0 ? 0 : x;
    int startY = y < 0 ? 0 : y;
    int endX = (x + w) > src.width ? src.width : (x + w);
    int endY = (y + h) > src.height ? src.height : (y + h);

    img.Image cropped = img.Image(width: endX - startX, height: endY - startY);
    for (int i = 0; i < cropped.height; i++) {
      for (int j = 0; j < cropped.width; j++) {
        cropped.setPixel(j, i, src.getPixel(startX + j, startY + i));
      }
    }

    return cropped;
  }

  Uint8List preprocessFace(CameraImage image, int x, int y, int w, int h) {
    final img.Image originalImage = convertCameraImageToImage(image);
    final img.Image face =
        copyCrop(originalImage, x - 5, y - 5, w + 10, h + 10);
    final img.Image resizedFace = img.copyResize(face, width: 160, height: 160);
    final Float32List input = imageToByteList(resizedFace, 160);
    return input.buffer.asUint8List();
  }

  img.Image convertCameraImageToImage(CameraImage image) {
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

        rgbImage.setPixel(x, y, getColor(r, g, b) as img.Color);
      }
    }

    return rgbImage;
  }

  int getColor(int r, int g, int b) {
    return (0xFF << 24) | (r << 16) | (g << 8) | b;
  }

  Float32List imageToByteList(img.Image image, int inputSize) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (int i = 0; i < inputSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        int pixel = image.getPixel(j, i) as int;
        buffer[pixelIndex++] = ((pixel >> 16) & 0xFF) / 255.0; // Red
        buffer[pixelIndex++] = ((pixel >> 8) & 0xFF) / 255.0; // Green
        buffer[pixelIndex++] = (pixel & 0xFF) / 255.0; // Blue
      }
    }
    return buffer;
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
      drawFocusCorners(canvas, rect, color);
    }
  }

  void drawFocusCorners(Canvas canvas, Rect rect, Color color) {
    final double cornerLength = 20.0;
    final double thickness = 3.0;

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = thickness;

    final corners = [
      // Top-left corner
      [
        Offset(rect.left, rect.top),
        Offset(rect.left + cornerLength, rect.top),
        Offset(rect.left, rect.top + cornerLength),
      ],
      // Top-right corner
      [
        Offset(rect.right, rect.top),
        Offset(rect.right - cornerLength, rect.top),
        Offset(rect.right, rect.top + cornerLength),
      ],
      // Bottom-left corner
      [
        Offset(rect.left, rect.bottom),
        Offset(rect.left + cornerLength, rect.bottom),
        Offset(rect.left, rect.bottom - cornerLength),
      ],
      // Bottom-right corner
      [
        Offset(rect.right, rect.bottom),
        Offset(rect.right - cornerLength, rect.bottom),
        Offset(rect.right, rect.bottom - cornerLength),
      ],
    ];

    for (var corner in corners) {
      canvas.drawLine(corner[0], corner[1], paint);
      canvas.drawLine(corner[0], corner[2], paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
