const fs = require('fs');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');

const modelOptions = {
  modelPath: 'file://model/anti-spoofing.json',
  outputTensors: ['activation_4'],
};

// load image from file and prepares image tensor that fits the model
async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data); // create rgb tensor from image
    const resize = tf.image.resizeBilinear(buffer, [inputSize, inputSize]); // model input resolution is 128x128
    const normalize = tf.div(resize, 255); // normalize input to range 0..1
    const expand = tf.expandDims(normalize, 0);
    const tensor = expand;
    const img = { fileName, tensor, inputShape: [buffer.shape[1], buffer.shape[0]], outputShape: tensor.shape, size: buffer.size };
    return img;
  });
  return obj;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);

  // load image and get approprite tensor for it
  const inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'outputShape:', img.outputShape);

  // run actual prediction
  const res = model.execute(img.tensor, modelOptions.outputTensors);
  const real = await res.data();
  log.data('Real?', real[0]);
}

main();
