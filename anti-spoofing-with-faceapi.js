const fs = require('fs');
const process = require('process');
const path = require('path');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const faceapi = require('@vladmandic/face-api');

const modelPathRoot = 'node_modules/@vladmandic/face-api/model';
const minConfidence = 0.5;
const fakeThreshold = 0.8;
const maxResults = 5;
let optionsSSDMobileNet;

async function main() {
  log.header();
  log.info('FaceAPI with AntiSpoofing');

  const input = process.argv[2];
  if (!input || !fs.existsSync(input)) {
    log.error('input image file missing');
    process.exit(1);
  }

  log.info('Initializing TFJS');
  await faceapi.tf.setBackend('tensorflow');
  await faceapi.tf.ready();

  log.info('Loading FaceAPI models');
  const modelPath = path.join(__dirname, modelPathRoot);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.ageGenderNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceExpressionNet.loadFromDisk(modelPath);
  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({ minConfidence, maxResults });

  log.info('Loading Anti-Spoofing model');
  const antispoofing = await tf.loadGraphModel('file://model-graph-f16/anti-spoofing.json');
  const inputSize = Object.values(antispoofing.modelSignature['inputs'])[0].tensorShape.dim[2].size;

  const buffer = fs.readFileSync(input);
  const decode = tf.node.decodeImage(buffer, 3);
  const tensor = tf.expandDims(decode, 0);
  tf.dispose(decode);
  log.info('Loaded image:', input, tensor.shape);

  log.info('Running FaceAPI detection');
  const result = await faceapi
    .detectAllFaces(tensor, optionsSSDMobileNet)
    .withFaceLandmarks();

  for (const face of result) {
    const box = [face.alignedRect.box.x, face.alignedRect.box.y, face.alignedRect.box.width, face.alignedRect.box.height].map((a) => Math.round(a));
    log.data(`Face: ${Math.round(100 * face.detection.score)}% confidence`, 'Box:', box);

    const score = tf.tidy(() => {
      const cropBox = [box[1] / tensor.shape[1], box[0] / tensor.shape[2], (box[3] + box[1]) / tensor.shape[1], (box[2] + box[0]) / tensor.shape[2]];
      log.info('Running Anti-Spoofing detection on cropped image:', cropBox);
      const cropped = tf.image.cropAndResize(tensor, [cropBox], [0], [inputSize, inputSize]);
      const norm = tf.div(cropped, 255);
      const res = antispoofing.execute(norm);
      return res.dataSync()[0];
    });
    log.state(`Real or Fake? ${score > fakeThreshold ? 'Fake' : 'Real'} (${Math.round(100 * score)}%)`);
  }
  tf.dispose(tensor);
}

main();
