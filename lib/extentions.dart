// import 'dart:async';
// import 'dart:typed_data';
//
// import 'package:camera/camera.dart';
// import 'package:flutter/material.dart';
// import 'package:flutter/services.dart';
// import 'package:image/image.dart' as img;
//
// extension XFileToCameraImage on XFile {
//   Future<CameraImage?> toCameraImage() async {
//     try {
//       final imageBytes = await readAsBytes();
//
//       final img.Image? image = img.decodeImage(imageBytes);
//
//       final int width = image?.width ?? 0;
//       final int height = image?.height ?? 0;
//
//       final yuvPlanes = await _convertImageToYUVPlane(image);
//
//       return CameraImage.fromBytes(
//         yuvPlanes[0],
//         yuvPlanes[1],
//         width,
//         height,
//         CameraImageFormat.yuv420,
//         planeRowStride: width,
//         planeRightRowStride: width,
//       );
//     } catch (error) {
//       print('Error converting XFile to CameraImage: $error');
//       return null;
//     }
//   }
//
//   Future<List<Uint8List>> _convertImageToYUVPlane(img.Image? image) async {
//     final imgList = img.decodeImage(image.bytes);
//     if (imgList.planes.length != 3) {
//       throw Exception(
//           'Image format not supported (expected YUV420 with 3 planes)');
//     }
//     return [imgList.planes[0], imgList.planes[1]];
//   }
// }
