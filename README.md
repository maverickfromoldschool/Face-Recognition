What is this Project

This is the Face Recognition System using Deep learning and various other tools of machine learning implemented in Python.
The purpose of this project is to detect and recognize faces appear in the image, for this we build various pipeline. Each pipeline perform its individual task and pass its output to next pipeline as input. we divide this project itno mainly 3 pipeline.
1st pipeline is use to preprocess the dataset,here the dataset means the collection of images of each person on which the system needs to be trained. preprocessing the data means detect the faces and align them if they are not aligned.
2nd pipeline is use to train the machine learning for prediction of faces. In this pipeline the preprocess dataset goes in and what we get is the vector embedding for each character.
3rd pipeline is to test the test-image.this pipeline take single image as input then detect all faces from image then align them if they are not and finally compare each face with the faces of trained character and predict the result with certain acurracy.

what are prerequiste of this project

Good command in Python 3.x and basic knowledge of machine learning,deep learning and other stuff like tensorflow.

what are dependencies of project

i implement this system in ubuntu you can also go with Mac/Win.i choose ubuntu because setting up with dependencies is quite easy with ubuntu than with windows.
you need the following in order to work along with me
1)python 3.x
2)tensorflow 
3)dlib
These are the main dependency rest are easy to install via pip command, pip command to install package in python is given below

pip install "somepackage name".

To install dlib you can follow the https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only/
setting up with dlib is the most tough part part of project especially in windows, after managing the dlib setup rest is a piece of cake.

What is the structure of project,
structure of project can be anything but what i will recommend you is
    
   
├── etc
│ ├── 20170511–185253
│ │ ├── 20170511–185253.pb
├── data
├── medium_facenet_tutorial
│ ├── align_dlib.py
│ ├── download_and_extract_model.py
│ ├── __init__.py
│ ├── lfw_input.py
│ ├── preprocess.py
│ ├── shape_predictor_68_face_landmarks.dat
│ └── train_classifier.py
├── requirements.txt

WHAT WILL BE THE APPROACH 

In 2015, researchers from Google released a paper, FaceNet, which uses a convolutional neural network relying on the image pixels as the features, rather than extracting them manually. It achieved a new record accuracy of 99.63% on the LFW dataset.
FaceNet: In the FaceNet paper, a convolutional neural network architecture is proposed. For a loss function, FaceNet uses “triplet loss”. Triplet loss relies on minimizing the distance from positive examples, while maximizing the distance from negative examples.
Conceptually, this makes sense. Faces of the same identity should appear closer to each other than faces of another identity.
The important take away from the paper is the idea of representing a face as a 128-dimensional embedding. An embedding is the collective name for mapping input features to vectors. In a facial recognition system, these inputs are images containing a subject’s face, mapped to a numerical vector representation.Since these vector embeddings are represented in shared vector space, vector distance can be used to calculate the similarity between two vectors. In a facial recognition context, this can vector distance be applied to calculate how similar two faces are. Additionally, these embeddings can be used as feature inputs into a classification, clustering, or regression task.

This project is divided into 3 part
    DATA PREPROCESSING
    TRAINING THE MODEL
    TESTING THE MODEL
    
Data Preprocessing : Below, you’ll pre-process the images before passing them into the FaceNet model. Image pre-processing in a facial recognition context typically solves a few problems. These problems range from lighting differences, occlusion, alignment, segmentation. Below, you’ll address segmentation and alignment.First, you’ll solve the segmentation problem by finding the largest face in an image. This is useful as our training data does not have to be cropped for a face ahead of time.Second, you’ll solve alignment. In photographs, it is common for a face to not be perfectly center aligned with the image. To standardize input, you’ll apply a transform to center all images based on the location of eyes and bottom lip.First download dlib’s face landmark predictor,to process the dataset, you can download it from here.

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

After downloading and extracting paste the "Shapepredictor68landmark.dat" file in the project directory according to project structure. 
You’ll use this face landmark predictor to find the location of the inner eyes and bottom lips of a face in an image. These coordinates will be used to center align the image.

Now follow the Align_Dlib.py from the repostery

This file, sourced from CMU, provides methods for detecting a face in an image, finding facial landmarks, and alignment given these landmarks.

After this follow the preprocess.py from repostery

Now you created a preprocessor for your dataset. This file will read each image into memory, attempt to find the largest face, center align, and write the file to output. If a face cannot be found in the image, logging will be displayed to console with the filename.

Now that you’ve created a pipeline, time to get results. As the script supports parallelism, you will see increased performance by running with multiple cores. You’ll need to run the preprocessor in the docker environment to have access to the installed libraries.
Below, you’ll mount your project directory as a volume inside the docker container and run the preprocessing script on your input data. The results will be written to a directory specified with command line arguments.

what we have done yet in DATA Preprocessing

Using Dlib, you detected the largest face in an image and aligned the center of the face by the inner eyes and bottom lip. This alignment is a method for standardizing each image for use as feature input.

NOW TIME TO TRAIN THE MODEL ON PROCESSED DATA

TRAIN THE MODEL : Now that you’ve preprocessed the data, you’ll generate vector embeddings of each identity. These embeddings can then be used as input to a classification, regression or clustering task.You’ll use the Inception Resnet V1 as your convolutional neural network. First, create a file to download the weights to the model.By using pre-trained weights, you are able to apply transfer learning to a new dataset.to download the model run the following python script , by the help of this you will be able to use pretrained network.

Run "downloadandextractmodel.py"



 
