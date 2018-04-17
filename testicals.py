import tensorflow as tf
from tensorflow.python.framework import ops

from tensorflow.python.platform import gfile
import numpy as np
import pickle,operator
import cv2
import dlib,os
import Align_Dlib



def read_image_from_disk(filename_to_label_tuple):
    """
    Consumes input tensor and loads image
    :param filename_to_label_tuple:
    :type filename_to_label_tuple: list
    :return: tuple of image and label
    """
    label = filename_to_label_tuple[1]
    file_contents = tf.read_file(filename_to_label_tuple[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label







imgDim = 180
scale = 1.0
facePredictor = r"/home/deeplearning/PycharmProjects/asitis/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(facePredictor)

filename = [r"/home/deeplearning/Documents/input_img/char/akshay-and-jan.jpeg"]
output_path = r"/home/deeplearning/Documents/ghanta/"
image_name = os.path.basename(filename[0])[:-5]
print(image_name)

copy_image=cv2.imread(filename[0])
image = cv2.imread(filename[0] )
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


faces_locations =detector(image,3)


points = [ predictor(image, faces_locations[x]) for x in range(len(faces_locations))]
landmarks = [[(p.x, p.y) for p in points[i].parts()] for i in range(len(faces_locations))]
npLandmarks = [np.float32(landmarks[x]) for x in range(len(landmarks))]

npLandmarkIndices = np.array(Align_Dlib.INNER_EYES_AND_BOTTOM_LIP)


H = [cv2.getAffineTransform(npLandmarks[x][npLandmarkIndices],
                            imgDim * Align_Dlib.MINMAX_TEMPLATE[npLandmarkIndices] * scale + imgDim * (1 - scale) / 2) for x in range(len(npLandmarks))]


thumbnail = [cv2.warpAffine(image, H[x], (imgDim, imgDim)) for x in range(len(H))]
thumbnail = [cv2.cvtColor(thumbnail[x], cv2.COLOR_BGR2RGB) for x in range(len(thumbnail))]



final_directory = [output_path+image_name+ str(int(x))+".jpeg" for x in range(len(thumbnail))]
print(final_directory)

for x in range(len(thumbnail)):
    cv2.imwrite(final_directory[x],thumbnail[x])


batch_size = len(final_directory)
for i,x in enumerate(faces_locations):
    copy_image = cv2.rectangle(image,(x.left(),x.top()),(x.right(),x.bottom()),color=(0,255,0))





all_label = [0,1,2,3,4,5,6,7,8,9,10,11]

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

    images = ops.convert_to_tensor(final_directory, dtype=tf.string)
    labels = ops.convert_to_tensor(all_label, dtype=tf.int32)


    input_queue = tf.train.slice_input_producer((images, labels),
                                                num_epochs=5, shuffle=False, )

    images_labels = []
    imgs = []
    lbls = []

    image, label = read_image_from_disk(filename_to_label_tuple=input_queue)
    image = tf.random_crop(image, size=[180, 180, 3])
    image.set_shape((180, 180, 3))
    image = tf.image.per_image_standardization(image)


    imgs.append(image)
    lbls.append(label)
    images_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(images_labels,
                                                   batch_size=batch_size,
                                                   capacity= 4 ,
                                                   enqueue_many=False,
                                                   allow_smaller_final_batch=True)

    images = image_batch
    labels = label_batch
    model_filepath = r"/home/deeplearning/PycharmProjects/asitis/etc/20170511-185253/20170511-185253.pb"

    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        print("jay bhavani\n")
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    emb_array = None
    label_array = None
    batch_images, batch_labels = sess.run([images, labels])


    emb = sess.run(embedding_layer,
                   feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})
    emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb
    label_array = np.concatenate([label_array, batch_labels]) if label_array is not None else batch_labels
    coord.request_stop()
    coord.join(threads=threads)
    classifier_filename = r"/home/deeplearning/PycharmProjects/letmetry/another_classifier.pkl"



    with open(classifier_filename, 'rb') as f:
        model, class_names = pickle.load(f)


        predictions = model.predict_proba(emb_array, )

        best_class_indices = np.argmax(predictions, axis=1)
        print(best_class_indices)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        personAndProbability = []
        for x in range(0, len(best_class_probabilities), 4):
            print(max(enumerate(best_class_probabilities[x:x + 4]), key=operator.itemgetter(1)))
            idx, vlu = max(enumerate(best_class_probabilities[x:x + 4]), key=operator.itemgetter(1))
            idx = idx + x
            idx = best_class_indices[idx]
            person = class_names[idx]
            personAndProbability.append((person,vlu))
        print(personAndProbability)

        person_dictionary = {
            'akshay': 0,
            'hawkeye': 0,
            'kriti': 0,
            'musk': 0,
            'peta jenson': 0,
            'randy': 0,
            'ronaldo': 0,
            'salman': 0,
            'shahruk': 0,
            'thor': 0,
            'tony': 0,
            'virat':0

            }
        for idx,(p,pr) in zip(range(len(personAndProbability)),personAndProbability):
            if pr>=0.5:
                person_dictionary[p] += 1
                rect = faces_locations[idx]
                cv2.putText(copy_image, p, (faces_locations[idx].left(), faces_locations[idx].top()), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        copy_image =cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("kuch bhi",copy_image)

        cv2.waitKey()

        print(person_dictionary)

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))


