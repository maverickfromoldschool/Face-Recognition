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

What is the structure of project
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
