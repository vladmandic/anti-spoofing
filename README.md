# Anti-Spoofing

Very simple sequential model trained on real/fake face images dataset published on Kaggle (1k real and 1k fake)  
Resulting quantized TFJS graph model is just ~800KB  

- Input is image `[1, 128, 128, 3]`  
- Output is `1` (real) or `0` (fake)

<br><hr><br>

## Model Architecture

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

<br><hr><br>

## Test Model

```shell
node anti-spoofing.js train/me.jpg
```

```js
2021-06-15 09:21:57 INFO:  anti-spoofing version 0.0.1
2021-06-15 09:21:57 INFO:  User: vlado Platform: linux Arch: x64 Node: v16.2.0
2021-06-15 09:21:57 INFO:  Loaded model { modelPath: 'file://model-graph-f16/anti-spoofing.json', outputTensors: [ 'activation_4', [length]: 1 ] } tensors: 11 bytes: 1706188
2021-06-15 09:21:57 INFO:  Loaded image: train/me.jpg inputShape: [ 1536, 2048, [length]: 2 ] outputShape: [ 1, 128, 128, 3, [length]: 4 ]
2021-06-15 09:21:57 DATA:  Real? true
```

<br><hr><br>

## Create, Train & Convert Model

### Create & Train

- [Jupyter Notebook](anti-spoofing.ipynb)

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

<br><hr><br>

## Links & Credits

- [Training data](https://www.kaggle.com/ciplab/real-and-fake-face-detection)
- [Model definition](https://www.kaggle.com/anku420/fake-face-detection/)
