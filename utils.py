import numpy as np
import os
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
from glob import glob
from tqdm import tqdm

class Image_data:

    def __init__(self, img_height, img_width, channels, dataset_path, label_list, augment_flag):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.augment_flag = augment_flag

        self.label_list = label_list
        self.dataset_path = dataset_path

        self.label_onehot_list = []
        self.image = []
        self.label = []


    def image_processing(self, filename, label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize_images(x_decode, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 127.5 - 1


        if self.augment_flag :
            augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            img = tf.cond(pred=tf.greater_equal(tf.random_uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda : augmentation(img, augment_height_size, augment_width_size),
                          false_fn=lambda : img)

        return img, label

    def preprocess(self):
        # self.label_list = ['tiger', 'cat', 'dog', 'lion']

        v = 0

        for label in self.label_list : # fabric
            label_one_hot = list(get_one_hot(v, len(self.label_list))) # [1, 0, 0, 0, 0]
            self.label_onehot_list.append(label_one_hot)
            v = v+1

            image_list = glob(os.path.join(self.dataset_path, label) + '/*.png') + glob(os.path.join(self.dataset_path, label) + '/*.jpg')
            label_one_hot = [label_one_hot] * len(image_list)

            self.image.extend(image_list)
            self.label.extend(label_one_hot)

class ImageData_celebA:

    def __init__(self, img_height, img_width, channels, dataset_path, label_list, augment_flag):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.augment_flag = augment_flag
        self.label_list = label_list

        self.dataset_path = os.path.join(dataset_path, 'train')
        self.file_name_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*.png')]
        self.lines = open(os.path.join(dataset_path, 'list_attr_celeba.txt'), 'r').readlines()

        self.image = []
        self.label = []

        self.test_image = []
        self.test_label = []

        self.attr2idx = {}
        self.idx2attr = {}

        self.train_label_onehot_list = []
        self.test_label_onehot_list = []

    def image_processing(self, filename, label, fix_label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize_images(x_decode, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_height = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            img = tf.cond(pred=tf.greater_equal(tf.random_uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda: augmentation(img, augment_height, augment_width),
                          false_fn=lambda: img)


        return img, label, fix_label

    def preprocess(self, phase):

        all_attr_names = self.lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = self.lines[2:]
        random.seed(1234)
        random.shuffle(lines)

        for i, line in enumerate(tqdm(lines)):
            split = line.split()
            if split[0] in self.file_name_list:
                filename = os.path.join(self.dataset_path, split[0])
                values = split[1:]

                label = []

                for attr_name in self.label_list:
                    idx = self.attr2idx[attr_name]

                    if values[idx] == '1':
                        label.append(1.0)
                    else:
                        label.append(0.0)

                if i < 2000:
                    self.test_image.append(filename)
                    self.test_label.append(label)
                else:
                    if phase == 'test' :
                        break
                    self.image.append(filename)
                    self.label.append(label)
                # ['./dataset/celebA/train/019932.png', [1, 0, 0, 0, 1]]

        print()

        self.test_label_onehot_list = create_labels(self.test_label, self.label_list)
        if phase == 'train' :
            self.train_label_onehot_list = create_labels(self.label, self.label_list)

        print('\n Finished preprocessing the CelebA dataset...')

def load_test_image(image_path, img_width, img_height, img_channel):

    if img_channel == 1 :
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1 :
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else :
        img = np.expand_dims(img, axis=0)

    img = img/127.5 - 1

    return img

def load_one_hot_vector(label_list, target_label) :
    label_onehot_dict = {}

    v = 0
    for label_name in label_list :
        label_one_hot = list(get_one_hot(v, len(label_list)))
        label_onehot_dict[label_name] = [label_one_hot]

    x = label_onehot_dict[target_label]


    return x

def label2onehot(label_list) :
    v = 0
    label_onehot_list = []
    for _ in label_list:  # fabric
        label_one_hot = list(get_one_hot(v, len(label_list)))  # [1, 0, 0, 0, 0]
        label_onehot_list.append(label_one_hot)
        v = v + 1

    return label_onehot_list

def augmentation(image, augment_height, augment_width):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_height, augment_width])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0


def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def return_images(images, size) :
    x = merge(images, size)

    return x

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def get_one_hot(targets, nb_classes):

    x = np.eye(nb_classes)[targets]

    return x

def create_labels(c_org, selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    c_org = np.asarray(c_org)
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []

    for i in range(len(selected_attrs)):
        c_trg = c_org.copy()

        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1.0
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0.0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

        c_trg_list.append(c_trg)

    c_trg_list = np.transpose(c_trg_list, axes=[1, 0, 2]) # [bs, c_dim, c_dim]

    return c_trg_list

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    if uniform :
        factor = gain * gain
        mode = 'FAN_AVG'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_AVG'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='relu', uniform=False) :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function =='tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    if uniform :
        factor = gain * gain
        mode = 'FAN_IN'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_IN'

    return factor, mode, uniform