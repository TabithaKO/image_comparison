# image_comparison
The repo. includes source code for image comparison using Machine Learning. (Triplet Loss)

The project demonstrates the use of convolutional netowork (CNN) trained using triplet loss. Triplet Loss minimises the distance between an anchor and a positive, images that contain same identity, and maximises the distance between the anchor and a negative, images that contain different identities.<br />
Conceptually it means, similar looking images should appear closer to each other than images with different looking view.

### Motivation
There is a business problem in which a survilleance camera is set in a particular field of view. However someone else would change its view which needs to be notified.<br />
The idea is to use the CNN with Triplet loss to try and solve the problem. Maintaining an reference image for each camera view in a database which can be tagged as the desired view.<br />
During run time, the camera view will generate and embedding and compare it using cosine similarity with all the embeddings(of camera view) present in the database. Therefore by thresholding, we can reach to a conclusion on whether the camera view has been changed or not.

### Requirements
```
Linux (Tested on MacOS)
Python
Python Packages
 - numpy
 - requests
 - TensorFlow
 - matplotlib
 ```
 You can use the `pip install <package_name>` command to solve the above python dependencies.
 
 The repo has the direcotry structure as depicted below:
```
image_comparison
├─ model.py
├─ preprocessing.py
├─ setup.sh
├─ train_triplets.py
├─ prediction.py
├─ data_repository
│   ├─ test
│   └─ train
├─ download_dataset.py
├─ requirements.txt
└─ README.md
```
- `model.py` file contains the source code for network architecture. The netowrk is a small CNN. It has 5 layers of (convolution + Pooling) layers followed by flattening of the vector. It takes the input of shape 28x28x3. Shape of the output embedding is 784 1-Dimensional
- `preprocessing.py` file contains source code for reading the dataset into numpy arrays, shuffling the images, normalizing it and splitting it into train and test sets. Also it picks a positive sample and a negative sample randomly for each training iteration.
- `train_triplets.py` file contains the source code for initiating the training. Various hyperparameter values are set in it.
- `download_dataset.py` downloads the zip file and unzip the data in desired directory i.e inside `data_repository`
- `setup.sh` installs the required python dependencies and sets up the envoirnment for training and inference.

### Training
Download Training Dataset by executing download_dataset.py
```
python download_dataset.py
```

To train
```
python train_triplets.py 
```
### Inference
```
python prediction.py
```
