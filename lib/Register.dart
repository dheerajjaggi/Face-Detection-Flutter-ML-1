import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:ui';
import 'package:face_detection/Database.dart';
import 'package:quiver/collection.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as imglib;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'main.dart';

class RegisterFace extends StatefulWidget {
  const RegisterFace({super.key});

  @override
  State<RegisterFace> createState() => _RegisterFaceState();
}

class _RegisterFaceState extends State<RegisterFace> {
  final TextEditingController _number = TextEditingController();
  final TextEditingController _txtUpiid = TextEditingController();
  // final TextEditingController _upiId = TextEditingController();
  var interpreter;
  late String _upiId; // Variable to store UPI ID
  late String _payee;
  List? e1;
  bool _fetchingData = false; // Add a variable to track data fetching process

  Directory? tempDir;
  File? jsonFile;
  bool isAlertBoxOpen = false;

  bool _faceFound = false;
  dynamic data = {};
  dynamic controller;
  bool isBusy = false;
  dynamic faceDetector;
  double threshold = 1.0;
  late Size size;
  late List<Face> faces;
  late CameraDescription description = cameras[1];
  CameraLensDirection camDirec = CameraLensDirection.front;

  @override
  void initState() {
    super.initState();
    initializeCamera();
    loadModel().then((_) {
      print("Model loaded successfully.");
      // Ensure that the interpreter is not null before proceeding
      if (interpreter != null) {
        print("Interpreter is ready.");
        // Proceed with operations that require the interpreter
      } else {
        print("Failed to load the model.");
        // Handle the error appropriately
      }
    });
  }

  @override
  void dispose() {
    isAlertBoxOpen = false;

    controller?.dispose();
    faceDetector.close();
    super.dispose();
  }

  initializeCamera() async {
    // if (controller != null) {
    //   await controller!.dispose();
    // }
    // if (faceDetector != null) {
    //   await faceDetector.close();
    // }
    // Initialize detector
    isAlertBoxOpen = false;
    final options =
        FaceDetectorOptions(enableContours: true, enableLandmarks: true);
    faceDetector = FaceDetector(options: options);

    controller = CameraController(description, ResolutionPreset.high);
    await controller.initialize();

    if (!mounted) {
      return;
    }

    controller!.startImageStream((image) {
      if (!isBusy) {
        isBusy = true;
        img = image;
        doFaceDetectionOnFrame();
      }
    });
  }

  //close all resources

  //face detection on a frame
  dynamic _scanResults;
  CameraImage? img;
  Future loadModel() async {
    try {
      this.interpreter =
          await tfl.Interpreter.fromAsset('mobilefacenet.tflite');

      print(
          '**********\n Loaded successfully model mobilefacenet.tflite \n*********\n');
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }
  }

  

  doFaceDetectionOnFrame() async {
    if (!isAlertBoxOpen) {
      // dynamic finalResult = Multimap<String, Face>();
      var frameImg = getInputImage();
      List<Face> faces = await faceDetector.processImage(frameImg);
      imglib.Image convertedImage = _convertCameraImage(img!, camDirec);
      if (faces.length == 0)
        _faceFound = false;
      else
        _faceFound = true;
      for (Face face in faces) {
        // Calculate the bounding box with a margin
        double x = face.boundingBox.left - 10;
        double y = face.boundingBox.top - 10;
        double w = face.boundingBox.width + 10;
        double h = face.boundingBox.height + 10;
        imglib.Image croppedImage = imglib.copyCrop(
          convertedImage,
          x.round(),
          y.round(),
          w.round(),
          h.round(),
        );

        // Resize the cropped image if necessary
        croppedImage = imglib.copyResizeCropSquare(croppedImage, 112);

        // Run the face recognition model on the processed image
        e1 = _recog(croppedImage);
        // print(_recog(croppedImage));
        // finalResult.add(res, face);
      }

      setState(() {
        // _scanResults = faces;
        _scanResults = faces;
        isBusy = false;
      });
    }
  }

  imglib.Image _convertCameraImage(
      CameraImage image, CameraLensDirection _dir) {
    int width = image.width;
    int height = image.height;
    // imglib -> Image package from https://pub.dartlang.org/packages/image
    var img = imglib.Image(width, height); // Create Image buffer
    const int hexFF = 0xFF000000;
    final int uvyButtonStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        final int uvIndex =
            uvPixelStride * (x / 2).floor() + uvyButtonStride * (y / 2).floor();
        final int index = y * width + x;
        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        // Calculate pixel color
        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
            .round()
            .clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
        // color: 0x FF  FF  FF  FF
        //           A   B   G   R
        img.data[index] = hexFF | (b << 16) | (g << 8) | r;
      }
    }
    var img1 = (_dir == CameraLensDirection.front)
        ? imglib.copyRotate(img, -90)
        : imglib.copyRotate(img, 90);
    return img1;
  }

  List _recog(imglib.Image img) {
    List input = imageToByteListFloat32(img, 112, 128, 128);
    input = input.reshape([1, 112, 112, 3]);
    List output = List.filled(1 * 192, null, growable: false).reshape([1, 192]);
    interpreter.run(input, output);
    output = output.reshape([192]);
    e1 = List.from(output);
    // return compare(e1!).toUpperCase();
    return e1!;
    // return e1!;
  }

  String compare(List currEmb) {
    if (data.length == 0) return "No Face saved";
    double minDist = 999;
    double currDist = 0.0;
    String predRes = "NOT RECOGNIZED";
    for (String label in data.keys) {
      currDist = euclideanDistance(data[label], currEmb);
      if (currDist <= threshold && currDist < minDist) {
        minDist = currDist;
        predRes = label;
      }
    }
    print(minDist.toString() + " " + predRes);
    return predRes;
  }

  double euclideanDistance(List e1, List e2) {
    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += pow((e1[i] - e2[i]), 2);
    }
    return sqrt(sum);
  }

  Float32List imageToByteListFloat32(
      imglib.Image image, int inputSize, double mean, double std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  InputImage getInputImage() {
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in img!.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();
    final Size imageSize = Size(img!.width.toDouble(), img!.height.toDouble());
    final camera = description;
    final imageRotation =
        InputImageRotationValue.fromRawValue(camera.sensorOrientation);
    final inputImageFormat =
        InputImageFormatValue.fromRawValue(img!.format.raw);

    final inputImageData = InputImageMetadata(
      size: imageSize,
      rotation: imageRotation!,
      format: inputImageFormat!,
      bytesPerRow: img!.planes[0].bytesPerRow,
    );

    final inputImage = InputImage.fromBytes(
      bytes: bytes,
      metadata: inputImageData,
    );

    return inputImage;
  }

  //Show rectangles around detected faces
  Widget buildResult() {
    if (_scanResults == null ||
        controller == null ||
        !controller.value.isInitialized) {
      return const Text('');
    }

    final Size imageSize = Size(
      controller.value.previewSize!.height,
      controller.value.previewSize!.width,
    );
    CustomPainter painter =
        FaceDetectorPainter(imageSize, _scanResults, camDirec);
    return CustomPaint(
      painter: painter,
    );
  }

  //toggle camera direction
  void toggleCameraDirection() async {
    if (camDirec == CameraLensDirection.back) {
      camDirec = CameraLensDirection.front;
      description = cameras[1];
    } else {
      camDirec = CameraLensDirection.back;
      description = cameras[0];
    }
    await controller.stopImageStream();
    setState(() {
      controller;
    });

    initializeCamera();
  }

  void _handle(String mobileNumber, String email, List embedding, String payee,
      String upiId) async {
    // Convert the embedding list to a JSON string
    String embeddingJson = json.encode(embedding);

    // Create a map with column names as keys and the data as values
    Map<String, dynamic> row = {
      DatabaseHelper.columnMobileNum: mobileNumber,
      DatabaseHelper.columnEmailId: email, // Corrected column name
      DatabaseHelper.columnUpiId: upiId,
      DatabaseHelper.columnPayee: payee,
      DatabaseHelper.columnEmbedding: embeddingJson
    };

    // Insert the row into the database using the singleton instance
    final id = await DatabaseHelper()
        .insert(row); // Corrected to use the singleton instance
    print(
        'inserted row id:================================================================================== $id');
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Registration Success'),
          content: Text('Valid UpiId was found: $_upiId'),
          actions: <Widget>[
            TextButton(
              child: const Text('OK'),
              onPressed: () {
                setState(() {
                  controller;
                });
                initializeCamera(); // Call initializeCamera when OK is pressed
                Navigator.of(context).pop(); // Close the dialog
              },
            ),
          ],
        );
      },
    );
  }

  bool isValidMobileNumber(String mobileNumber) {
    // Check if the mobile number is not empty and has exactly 10 digits
    return mobileNumber.isNotEmpty && mobileNumber.length == 10;
  }



  void _addLabel() {
    setState(() {
      controller = null;
      isAlertBoxOpen = true;
    });
    bool agreeToTerms = false; // Track whether the user agrees to terms

    var alert = AlertDialog(
      title: const Text("Register"),
      content: StatefulBuilder(
        builder: (BuildContext context, StateSetter setState) {
          return Column(
            children: <Widget>[
              Expanded(
                child: TextField(
                  controller: _number,
                  autofocus: true,
                  style:
                      const TextStyle(fontSize: 12), // Adjust the font size as needed

                  decoration: const InputDecoration(
                    labelText: "Mobile Number",
                    labelStyle: TextStyle(
                        fontSize: 12), // Adjust the font size as needed

                    icon: Icon(Icons.phone),
                  ),
                ),
              ),
              Expanded(
                child: TextField(
                  controller: _txtUpiid,
                  autofocus: true,
                  style:
                      const TextStyle(fontSize: 12), // Adjust the font size as needed

                  decoration: const InputDecoration(
                    labelText: "UPI ID",
                    labelStyle: TextStyle(
                        fontSize: 12), // Adjust the font size as needed

                    icon: Icon(Icons.payment),
                  ),
                ),
              ),
              CheckboxListTile(
                title: const Text(
                  "I agree to provide my phone number and email address for receiving payments during the demonstration and for discussions during API Days.",
                  style:
                      TextStyle(fontSize: 10), // Adjust the font size as needed
                ),
                value: agreeToTerms,
                onChanged: (newValue) {
                  setState(() {
                    agreeToTerms = newValue!;
                  });
                },
                contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16, vertical: 8), // Add padding around the text

                controlAffinity: ListTileControlAffinity.leading,
              ),
            ],
          );
        },
      ),
      actions: <Widget>[
        TextButton(
            child: const Text("Save"),
            onPressed: () {
              if (!agreeToTerms) {
                // Show an alert if the user hasn't agreed to terms
                showDialog(
                  context: context,
                  builder: (BuildContext context) {
                    return AlertDialog(
                      title: const Text('Error'),
                      content:
                          const Text('Please agree to the terms and conditions.'),
                      actions: <Widget>[
                        TextButton(
                          child: const Text('OK'),
                          onPressed: () {
                            Navigator.of(context).pop();
                          },
                        ),
                      ],
                    );
                  },
                );
                return;
              }
              String mobileNumber = _number.text;
              String txtUpiId = _txtUpiid.text;

              if (!isValidMobileNumber(mobileNumber)) {
                showDialog(
                  context: context,
                  builder: (BuildContext context) {
                    return AlertDialog(
                      title: const Text('Error'),
                      content:
                          const Text('Please enter a valid mobile number.'),
                      actions: <Widget>[
                        TextButton(
                          child: const Text('OK'),
                          onPressed: () {
                            Navigator.of(context).pop();
                          },
                        ),
                      ],
                    );
                  },
                );
                return;
              }
          if (txtUpiId.isNotEmpty ) { 
            //Check for UPI ID           
                 checkUpiId(txtUpiId).then((upiFound) {
                if (upiFound) {                  
                  print("numberrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr$_number.text");
                  print("emailllllllllllllllllllllllllll$_txtUpiid.text");
                  // If UPI ID is found, proceed with the handle method
                  _handle(_number.text.toString(), _txtUpiid.text.toString(), e1!,
                      _payee, _upiId);
                }
                 } );
                
              }
              else{
           // _handle(_name.text.toUpperCase());
              fetchUpiIdMobile(_number.text).then((upiIdFound) {
                if (upiIdFound) {
                  print("numberrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr$_number.text");
                  print("emailllllllllllllllllllllllllll$_txtUpiid.text");
                  // If UPI ID is found, proceed with the handle method
                  _handle(_number.text.toString(), _txtUpiid.text.toString(), e1!,
                      _payee, _upiId);
                }
              });
              }

              
              Navigator.pop(context);
            }),
        TextButton(
          child: const Text("Cancel"),
          onPressed: () {
            initializeCamera();
            Navigator.pop(context);
          },
        )
      ],
    );
    showDialog(
        context: context,
        builder: (context) {
          return alert;
        });
  }

 Future<bool> checkUpiId(String upiId) async {
  setState(() {
      _fetchingData = true; // Set to true when data fetching begins
    });
    bool upiIdFound = await IsValidUpi(upiId);
     if (!upiIdFound) {
      setState(() {
        _upiId = 'invalid';
      });
      showDialog(
                  context: context,
                  builder: (BuildContext context) {
                    return AlertDialog(
                      title: const Text('Error'),
                      content: const Text('Please enter a valid UPI address.'),
                      actions: <Widget>[
                        TextButton(
                          child: const Text('OK'),
                          onPressed: () {
                            //Navigator.of(context).pop();                            
                          },
                        ),
                      ],
                    );
                  },
                );
     
      setState(() {
        _fetchingData = false; // Set to false when data fetching is complete
      });
     }
    print(_upiId);
    print(_payee);
    setState(() {
      _fetchingData = false; // Set to false when data fetching is complete
    });
    return upiIdFound; // Return true if UPI ID is found, false otherwise
 }
  Future<bool> fetchUpiIdMobile(String mobileNumber) async {
    setState(() {
      _fetchingData = true; // Set to true when data fetching begins
    });
    bool upiIdFound = await attemptFetchUpiIdWithMobileNumber(mobileNumber);
    if (!upiIdFound) {
      setState(() {
        _upiId = 'invalid';
      });
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Registration Successful'),
            content: Text('No UPI ID found for the provided mobile number.'),
            actions: <Widget>[
              TextButton(
                child: Text('OK'),
                onPressed: () {
                  setState(() {
                    controller;
                  });
                  Navigator.of(context).pop();
                  initializeCamera(); // Close the dialog
                },
              ),
            ],
          );
        },
      );
      setState(() {
        _fetchingData = false; // Set to false when data fetching is complete
      });
    }

    print(_upiId);
    print(_payee);
    setState(() {
      _fetchingData = false; // Set to false when data fetching is complete
    });
    return upiIdFound; // Return true if UPI ID is found, false otherwise
  }

  Future<bool> attemptFetchUpiIdWithMobileNumber(String mobileNumber) async {
    const upiExtensions = ["ybl", "axl", "ibl", "paytm"];
    String vpa = mobileNumber;
    for (final extension in upiExtensions) {
      final fullVpa = '$vpa@$extension';      
       if  (await IsValidUpi(fullVpa)) return true;
    }

    print("Attempting to fetch UPI ID using mobile number...");
    return false; // Placeholder return value
  }

  Future<bool> IsValidUpi(String fullVpa) async {
    const url =
        'https://upi-verification.p.rapidapi.com/v3/tasks/sync/verify_with_source/ind_vpa';
    final payload = {
      'task_id': 'UUID',
      'group_id': 'UUID',
      'data': {'vpa': fullVpa}
    };
    final headers = {
      'content-type': 'application/json',
      'X-RapidAPI-Key':
          '1b137fff14msh7add3a17ecf0b6bp14189djsn46c923e117cd', // Set your API key here
      'X-RapidAPI-Host': 'upi-verification.p.rapidapi.com'
    };
    try {
      final response = await http.post(Uri.parse(url),
          headers: headers, body: json.encode(payload));
      print(response.body);

      final Map<String, dynamic> data = json.decode(response.body);
      if (data['result'] != null && data['result']['status'] == 'id_found') {
        setState(() {
          _upiId = data['result']['vpa'];
          _payee = data['result']['name_at_bank'];
        });
        return true;
      }
    } catch (error) {
      print('Error fetching UPI ID: $error');
      return false;
    }
    return false;
  }

  

  @override
  Widget build(BuildContext context) {
    List<Widget> stackChildren = [];
    size = MediaQuery.of(context).size;
    if (controller != null) {
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          height: size.height - 250,
          child: Container(
            child: (controller.value.isInitialized)
                ? AspectRatio(
                    aspectRatio: controller.value.aspectRatio,
                    child: CameraPreview(controller),
                  )
                : Container(),
          ),
        ),
      );
      stackChildren.add(
        Positioned(
            top: 0.0,
            left: 0.0,
            width: size.width,
            height: size.height - 250,
            child: buildResult()),
      );
    }

    stackChildren.add(Positioned(
      top: size.height - 200,
      left: 0,
      width: size.width,
      height: 250,
      child: _fetchingData
          ? CircularProgressIndicator()
          : Container(
              color: const Color.fromARGB(255, 25, 24, 24),
              child: Center(
                child: Container(
                  margin: const EdgeInsets.only(bottom: 40),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          IconButton(
                            icon: const Icon(
                              Icons.cached,
                              color: Color.fromARGB(255, 252, 248, 248),
                            ),
                            iconSize: 50,
                            color: const Color.fromARGB(255, 204, 203, 203),
                            onPressed: () {
                              toggleCameraDirection();
                            },
                          )
                        ],
                      ),
                      FloatingActionButton(
                        backgroundColor:
                            (_faceFound) ? Colors.blue : Colors.blueGrey,
                        child: Icon(Icons.add),
                        onPressed: () {
                          if (_faceFound) _addLabel();
                        },
                        heroTag: null,
                      )
                    ],
                  ),
                ),
              ),
            ),
    ));

    return Scaffold(
      backgroundColor: Colors.black,
      body: Container(
          margin: const EdgeInsets.only(top: 0),
          color: const Color.fromARGB(255, 32, 31, 31),
          child: Stack(
            children: stackChildren,
          )),
    );
  }
}

class FaceDetectorPainter extends CustomPainter {
  FaceDetectorPainter(this.absoluteImageSize, this.faces, this.camDire2);

  final Size absoluteImageSize;
  final List<Face> faces;
  CameraLensDirection camDire2;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / absoluteImageSize.width;
    final double scaleY = size.height / absoluteImageSize.height;

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Color.fromARGB(255, 62, 230, 65);

    for (Face face in faces) {
      canvas.drawRect(
        Rect.fromLTRB(
          camDire2 == CameraLensDirection.front
              ? (absoluteImageSize.width - face.boundingBox.right) * scaleX
              : face.boundingBox.left * scaleX,
          face.boundingBox.top * scaleY,
          camDire2 == CameraLensDirection.front
              ? (absoluteImageSize.width - face.boundingBox.left) * scaleX
              : face.boundingBox.right * scaleX,
          face.boundingBox.bottom * scaleY,
        ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(FaceDetectorPainter oldDelegate) {
    return oldDelegate.absoluteImageSize != absoluteImageSize ||
        oldDelegate.faces != faces;
  }
}
