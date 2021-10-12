# Anti-Spoofing for TFJS

Very simple sequential model trained on real/fake face images dataset published on **Kaggle** (1k real and 1k fake)  
Resulting quantized **TensorFlow/JS Graph Model** is just **~800KB**  

- Input is cropped image of a face to analyze with shape `[1, 128, 128, 3]`  
- Output is value in range `0..1` where higher number means higher chance of real image

<br>

## Test Model: Standalone

> node anti-spoofing.js real.jpg  

```js
INFO:  anti-spoofing version 0.0.1
INFO:  Loaded model { modelPath: 'file://model-graph-f16/anti-spoofing.json', outputTensors: [ 'activation_4', [length]: 1 ] } tensors: 11 bytes: 1706188
INFO:  Loaded image: real.jpg inputShape: [ 1536, 2048, [length]: 2 ] outputShape: [ 1, 128, 128, 3, [length]: 4 ]
DATA:  Real? 0.9444
```

<br>

> node anti-spoofing.js fake.jpg  

```js
INFO:  anti-spoofing version 0.0.1
INFO:  Loaded model { modelPath: 'file://model-graph-f16/anti-spoofing.json', outputTensors: [ 'activation_4', [length]: 1 ] } tensors: 11 bytes: 1706188
INFO:  Loaded image: fake.jpg inputShape: [ 1536, 2048, [length]: 2 ] outputShape: [ 1, 128, 128, 3, [length]: 4 ]
DATA:  Real? 0.5234
```

<br>

## Test Model: with FaceAPI Detector

In case a non-cropped image is expected, best results are achieved by running a face detector before running anti-spoofing

> node anti-spoofing-with-faceapi.js fake.jpg  

```js
FaceAPI with AntiSpoofing
Loading FaceAPI models
Loading Anti-Spoofing model
Loaded image: fake.jpg [ 1, 1066, 800, 3, [length]: 4 ]
Running FaceAPI detection
Face: 74% confidence Box: [ 267, 205, 203, 137, [length]: 4 ]
Running Anti-Spoofing detection on cropped image: [ 0.19230769230769232, 0.33375, 0.32082551594746717, 0.5875, [length]: 4 ]
STATE: Real or Fake? Fake 93%
```

<br>

## Create, Train & Convert Model

### Create & Train

- [Jupyter Notebook](anti-spoofing.ipynb)

### Model Architecture

Extremely simple model...

```python
Conv2D(64,(3,3), input_shape=X_train.shape[1:]) 
MaxPooling2D(pool_size=(2,2))
Activation("relu") 
Conv2D(32,(3,3))  
MaxPooling2D(pool_size=(2,2))
Activation("relu")
Conv2D(16,(3,3)) 
MaxPooling2D(pool_size=(2,2))
Activation("relu")
Flatten() 
Dense(128,kernel_regularizer="l2")
Activation("relu",)
Dropout(0.12)
Dense(1) 
Activation("sigmoid")
compile(loss = "binary_crossentropy", optimizer = keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy'])
```

### Convert

```shell
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --quantize_float16=* --strip_debug_ops=* model-saved/ model-graph-f16/
```

### Model Signature

```js
2021-06-15 09:09:27 INFO:  anti-spoofing version 0.0.1
2021-06-15 09:09:27 INFO:  User: vlado Platform: linux Arch: x64 Node: v16.2.0
2021-06-15 09:09:27 DATA:  created on: 2021-06-15T12:55:48.792Z
2021-06-15 09:09:27 INFO:  graph model: /home/vlado/dev/anti-spoofing/model-graph-f16/model.json
2021-06-15 09:09:27 INFO:  size: { unreliable: true, numTensors: 11, numDataBuffers: 11, numBytes: 1706188 }
2021-06-15 09:09:27 INFO:  model inputs based on signature
2021-06-15 09:09:27 INFO:  model outputs based on signature
2021-06-15 09:09:27 DATA:  ops used by model: {
  graph: [ 'Const', 'Placeholder', 'Identity', [length]: 3 ],
  convolution: [ '_FusedConv2D', 'MaxPool', [length]: 2 ],
  basic_math: [ 'Relu', 'Sigmoid', [length]: 2 ],
  transformation: [ 'Reshape', [length]: 1 ],
  matrices: [ '_FusedMatMul', [length]: 1 ]
}
2021-06-15 09:09:27 DATA:  inputs: [ { name: 'conv2d_input', dtype: 'DT_FLOAT', shape: [ -1, 128, 128, 3, [length]: 4 ] }, [length]: 1 ]
2021-06-15 09:09:27 DATA:  outputs: [ { id: 0, name: 'activation_4', dytpe: 'DT_FLOAT', shape: [ -1, 1, [length]: 2 ] }, [length]: 1 ]
```

<br>

## Links & Credits

- [Training data](https://www.kaggle.com/ciplab/real-and-fake-face-detection)
- [Model definition](https://www.kaggle.com/anku420/fake-face-detection/)
