# Common config for the datasets custom parsers. If you change a configuration parameter
# as MAX_WIDTH, the setting will be common for the dataset specific parsers.
import cv2
import os
import re
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# TO CHANGE
CLASS_NUMBER = 43
classes_names = [
    "SPEED_LIMIT_20",
    "SPEED_LIMIT_30",
    "SPEED_LIMIT_50",
    "SPEED_LIMIT_60",
    "SPEED_LIMIT_70",
    "SPEED_LIMIT_80",
    "RESTRICTION_ENDS_80",
    "SPEED_LIMIT_100",
    "SPEED_LIMIT_120",
    "NO_OVERTAKING",
    "NO_OVERTAKING_TRUCKS",
    "PRIORITY_AT_NEXT_INTERSECTION",
    "PRIORITY_ROAD",
    "GIVE_WAY",
    "STOP",
    "NO_TRAFFIC_BOTH_WAYS",
    "NO_TRUCKS",
    "NO_ENTRY",
    "DANGER",
    "BEND_LEFT",
    "BEND_RIGHT",
    "BEND",
    "UNEVEN_ROAD",
    "SLIPPERY_ROAD",
    "ROAD_NARROWS",
    "CONSTRUCTION",
    "TRAFFIC_SIGNAL",
    "PEDESTRIAN_CROSSING",
    "SCHOOL_CROSSING",
    "CYCLES_CROSSING",
    "SNOW",
    "ANIMALS",
    "RESTRICTION_ENDS",
    "GO_RIGHT",
    "GO_LEFT",
    "GO_STRAIGHT",
    "GO_RIGHT_OR_STRAIGHT",
    "GO_LEFT_OR_STRAIGHT",
    "KEEP_RIGHT",
    "KEEP_LEFT",
    "ROUNDABOUT",
    "RESTRICTION_ENDS_OVERTAKING",
    "RESTRICTION_ENDS_OVERTAKING_TRUCKS",
]

OTHER_CLASS = CLASS_NUMBER + 1  # Class that will contain all the negative samples.
OTHER_CLASS_NAME = "other"


classes_counter_train = [0] * (CLASS_NUMBER + 1)
classes_counter_test = [0] * CLASS_NUMBER

TRAIN_PROB = 0.8
TEST_PROB = 0.2
ADD_FALSE_DATA = False

SHOW_IMG = False  # Show each image being processed (verbose)
COLOR_MODE = -1  # Color mode of the images read (-1 => RGB)
OUTPUT_IMG_EXTENSION = ".jpg"  # Output extension for the files processed.

# Prefix for each dataset parser. That way you can handle things different
# depending on the dataset from here.
DB_PREFIX = ""


# In the specific datasets, each object class has a different object id.
# Example: (speedlimit => 26 in BTSDB but speedlimit => 5 in GTSDB).
# For this reason, traffic_sign_classes structure creates a relation between
# the specific object id and the general one (common to all the datasets.)
traffic_sign_classes = {}


# Method that initialize the classes counter.
# (Necessary at the start of each datasets parser)
def initialize_classes_counter():
    for i in range(0, len(classes_counter_train)):
        classes_counter_train[i] = 0

    for i in range(0, len(classes_counter_test)):
        classes_counter_test[i] = 0


# Method that updates the db_prefix.
# (Necessary at the start of each datasets parser)
def update_db_prefix(db_prefix):
    global DB_PREFIX
    DB_PREFIX = db_prefix


# Reads the image using opencv.
def read_img(input_img_file_path):
    return cv2.imread(input_img_file_path, COLOR_MODE)


# Reads the image using plt.
def read_img_plt(input_img_file_path):
    return Image.open(input_img_file_path)


# Returns the image width and height for opencv image.
def get_img_dim(input_img):
    img_height, img_width, channels = input_img.shape
    return img_width, img_height


# Returns the image width and height for plt image.
def get_img_dim_plt(input_img):
    return input_img.size


# Write an opencv image (fastest).
def write_img(output_file_path, output_img):
    return cv2.imwrite(output_file_path + OUTPUT_IMG_EXTENSION, output_img)


# Returns an opencv image resized to MAX_WIDTH and MAX_HEIGHT.
def resize_img(input_img):
    return input_img
    # return cv2.resize(input_img, (MAX_WIDTH, MAX_HEIGHT))


def resize_img_percentage(input_img, percentage):
    image_width, image_height = get_img_dim(input_img)
    image_width = int(image_width * percentage)
    image_height = int(image_height * percentage)
    return cv2.resize(input_img, (image_width, image_height))


# Returns an plt image resized to MAX_WIDTH and MAX_HEIGHT.
def resize_img_plt(input_img, image_width, image_height):
    return np.asarray(input_img.resize((image_width, image_height)))


# Shows an input image with a bounding box using plt.
def show_img(img, object_lb_x1, object_lb_y1, object_width, object_height):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle(
        (object_lb_x1, object_lb_y1),
        object_width,
        object_height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.show()


# Returns a string with the darknet label for the received object_class,
# image dimensions and bounding box limits.
def parse_darknet_format(
    object_class, img_width, img_height, left_x, bottom_y, right_x, top_y
):
    object_width = right_x - left_x
    object_height = top_y - bottom_y
    object_mid_x = (left_x + right_x) / 2.0
    object_mid_y = (bottom_y + top_y) / 2.0

    object_width_rel = object_width / img_width
    object_height_rel = object_height / img_height
    object_mid_x_rel = object_mid_x / img_width
    object_mid_y_rel = object_mid_y / img_height

    dark_net_label = "{} {} {} {} {}".format(
        object_class,
        object_mid_x_rel,
        object_mid_y_rel,
        object_width_rel,
        object_height_rel,
    )

    return dark_net_label


# Saves the image received in the output file path in the OUTPUT_IMG_EXTENSION.
# Saves the filename in the general training/testing file.
# Saves the filename and darknet labels for each object in the txt file with the image filename.
def write_data(
    filename, input_img, input_img_labels, text_file, output_dir, train_file
):
    output_file_path = output_dir + filename

    # Save file in general training/testing file
    text_file.write(output_file_path + OUTPUT_IMG_EXTENSION + "\n")
    # Save file in correct folder
    write_img(output_file_path, input_img)

    # SAVE TXT FILE WITH THE IMG
    f = open(output_file_path + ".txt", "a")
    labels_to_print = ""
    for input_img_label in input_img_labels:
        labels_to_print += input_img_label + "\n"

        object_class = int(input_img_label.split()[0])
        if train_file:
            classes_counter_train[object_class] += 1
        else:
            classes_counter_test[object_class] += 1
    f.write(labels_to_print)


# Chooses a total number of total_false_negatives_count randomly from total_false_negatives_dir and
# saves them as background images in the training set.
def add_false_negatives(
    total_false_negatives_count,
    total_false_negatives_dir,
    output_train_dir_path,
    train_text_file,
):
    print("Adding " + str(total_false_negatives_count) + " false images...")
    false_negative_sublist = rand.sample(
        total_false_negatives_dir.keys(), total_false_negatives_count
    )

    for fn_filename in false_negative_sublist:
        fn_file_path = total_false_negatives_dir[fn_filename][0]
        input_img = read_img(fn_file_path)
        input_img = resize_img(input_img)

        if DB_PREFIX == "btsdb-":  # Specific output filename for parser
            output_filename = DB_PREFIX + "fn-" + fn_filename[3:-4]
        else:
            output_filename = DB_PREFIX + "fn-" + fn_filename[:-4]

        write_data(
            output_filename, input_img, [], train_text_file, output_train_dir_path, True
        )


# Prints the object's number of each class of the received array.
def print_class_info(classes_counter):
    for i in range(0, min(len(classes_names), len(classes_counter))):
        print('\t- CLASS ' + str(i) + " - " + classes_names[i] + ' : ' + str(classes_counter[i]))
    print('TOTAL: ' + str(sum(classes_counter)))


# Prints the train classes, test classes and proportion train-test for a DB.
def print_db_info(classes_counter_train, classes_counter_test):
    print("[TRAIN FILES]")
    print_class_info(classes_counter_train)

    print("\n[TEST FILES]")
    print_class_info(classes_counter_test)

    print("\n[PROPORTION]")
    for i in range(0, min(len(classes_counter_train), len(classes_counter_test))):
        total_classes = classes_counter_train[i] + classes_counter_test[i]
        if total_classes == 0:
            total_classes = 1
        print(
            "\t- CLASS "
            + str(i)
            + " - "
            + classes_names[i]
            + " : "
            + "{:.2f}%".format(classes_counter_test[i] / total_classes * 100.0)
        )


# Given two arrays, returns the sum of them.
def add_arrays(array_1, array_2):
    total_array = array_2.copy()

    for i in range(0, len(array_1)):
        total_array[i] += array_1[i]

    return total_array
