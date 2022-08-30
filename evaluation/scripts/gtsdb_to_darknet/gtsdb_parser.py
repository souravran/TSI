# Python program for converting the ppm files from The German Traffic Sign Recognition Benchmark (GTSRB) to jpg files
# in order to use them in YOLO. Besides, it generate a txt with all the paths to the converted images in darknet format.

import click
import csv
from gtsdb_config import *

# TO CHANGE
ROOT_PATH = os.path.expanduser("~/workspace/TSI/Data/gtsdb_darknet_test/")
GTSDB_SOURCE_PATH = os.path.expanduser("~/workspace/TSI/Data/FullIJCNN2013_darknet/")
RESIZE_PERCENTAGE = 0.6
DB_PREFIX = "gtsdb-"


ANNOTATIONS_FILE_PATH = GTSDB_SOURCE_PATH + "gt.txt"
INPUT_PATH = GTSDB_SOURCE_PATH  # Path to the ppm images of the GTSRB dataset.


def initialize_traffic_sign_classes():
    traffic_sign_classes = {
        "0_speed_limit_20": 0,
        "1_speed_limit_30": 1,
        "2_speed_limit_50": 2,
        "3_speed_limit_60": 3,
        "4_speed_limit_70": 4,
        "5_speed_limit_80": 5,
        "6_restriction_ends_80": 6,
        "7_speed_limit_100": 7,
        "8_speed_limit_120": 8,
        "9_no_overtaking": 9,
        "10_no_overtaking_trucks": 10,
        "11_priority_at_next_intersection": 11,
        "12_priority_road": 12,
        "13_give_way": 13,
        "14_stop": 14,
        "15_no_traffic_both_ways": 15,
        "16_no_trucks": 16,
        "17_no_entry": 17,
        "18_danger": 18,
        "19_bend_left": 19,
        "20_bend_right": 20,
        "21_bend": 21,
        "22_uneven_road": 22,
        "23_slippery_road": 23,
        "24_road_narrows": 24,
        "25_construction": 25,
        "26_traffic_signal": 26,
        "27_pedestrian_crossing": 27,
        "28_school_crossing": 28,
        "29_cycles_crossing": 29,
        "30_snow": 30,
        "31_animals": 31,
        "32_restriction_ends": 32,
        "33_go_right": 33,
        "34_go_left": 34,
        "35_go_straight": 35,
        "36_go_right_or_straight": 36,
        "37_go_left_or_straight": 37,
        "38_keep_right": 38,
        "39_keep_left": 39,
        "40_roundabout": 40,
        "41_restriction_ends_overtaking": 41,
        "42_restriction_ends_overtaking_trucks": 42,
    }
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = [
        43
    ]  # undefined, other classes


# It depends on the row format
def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = real_img_width / image_width
    height_proportion = real_img_height / image_height

    left_x = float(row[1]) / width_proportion
    bottom_y = float(row[2]) / height_proportion
    right_x = float(row[3]) / width_proportion
    top_y = float(row[4]) / height_proportion

    object_class = int(row[5])

    if SHOW_IMG:
        show_img(
            resize_img_plt(input_img, image_width, image_height),
            left_x,
            bottom_y,
            (right_x - left_x),
            (top_y - bottom_y),
        )

    return parse_darknet_format(
        object_class, image_width, image_height, left_x, bottom_y, right_x, top_y
    )


def update_global_variables(
    train_pct, test_pct, color_mode, verbose, false_data, output_img_ext
):
    global TRAIN_PROB, TEST_PROB, COLOR_MODE, SHOW_IMG, ADD_FALSE_DATA, OUTPUT_IMG_EXTENSION
    TRAIN_PROB = train_pct
    TEST_PROB = test_pct
    COLOR_MODE = color_mode
    SHOW_IMG = verbose
    ADD_FALSE_DATA = false_data
    OUTPUT_IMG_EXTENSION = output_img_ext


# Function for reading the images
def read_dataset(
    output_train_text_path,
    output_test_text_path,
    output_train_dir_path,
    output_test_dir_path,
):
    img_labels = {}  # Set of images and its labels [filename]: [()]
    update_db_prefix(DB_PREFIX)
    initialize_traffic_sign_classes()
    initialize_classes_counter()

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")

    gt_file = open(ANNOTATIONS_FILE_PATH)  # Annotations file
    gt_reader = csv.reader(gt_file, delimiter=";")  # CSV parser for annotations file

    # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
    for row in gt_reader:
        filename = row[0]
        file_path = INPUT_PATH + filename

        if os.path.isfile(file_path):
            input_img = read_img_plt(file_path)
            darknet_label = calculate_darknet_format(input_img, row)
            object_class_adjusted = int(darknet_label.split()[0])

            # If it is the first label for that img
            if (filename not in img_labels.keys()):  
                img_labels[filename] = [file_path]

            # Add only useful labels (not false negatives)
            if (object_class_adjusted != OTHER_CLASS):  
                img_labels[filename].append(darknet_label)

    # COUNT FALSE NEGATIVES (IMG WITHOUT LABELS)
    total_false_negatives_dir = {}
    total_annotated_images_dir = {}
    for filename in img_labels.keys():
        img_label_subset = img_labels[filename]
        if len(img_label_subset) == 1:
            total_false_negatives_dir[filename] = img_label_subset
        else:
            total_annotated_images_dir[filename] = img_label_subset

    total_annotated_images = len(img_labels.keys()) - len(total_false_negatives_dir.keys())
    total_false_negatives = len(total_false_negatives_dir.keys())
    max_false_data = round(total_annotated_images * TRAIN_PROB)  # False data: False negative + background

    print("total_false_negatives: " + str(total_false_negatives))
    print(
        "total_annotated_images: "
        + str(total_annotated_images)
        + " == "
        + str(len(total_annotated_images_dir.keys()))
    )
    print("max_false_data: " + str(max_false_data))

    # ADD FALSE IMAGES TO TRAIN
    if total_false_negatives > max_false_data:
        total_false_negatives = max_false_data

    if ADD_FALSE_DATA:
        add_false_negatives(total_false_negatives, total_false_negatives_dir, output_train_dir_path, train_text_file)

    for filename in total_annotated_images_dir.keys():
        input_img_file_path = img_labels[filename][0]
        input_img = read_img(input_img_file_path)  # Read image from image_file_path
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[filename][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + filename[:-4]

        if train_file:
            write_data(output_filename, input_img, input_img_labels, train_text_file, output_train_dir_path, train_file)
        else:
            write_data(output_filename, input_img, input_img_labels, test_text_file, output_test_dir_path, train_file)

    gt_file.close()
    train_text_file.close()
    test_text_file.close()

    return classes_counter_train, classes_counter_test


# Main method.
@click.command()
@click.option('--root_path', default=ROOT_PATH, help='Path where you want to save the dataset.')
@click.option('--train_pct', default=TRAIN_PROB, help='Percentage of train images in final dataset. Format (0.0 - 1.0)')
@click.option('--test_pct', default=TEST_PROB, help='Percentage of test images in final dataset. Format (0.0 - 1.0)')
@click.option('--color_mode', default=COLOR_MODE, help='OpenCV Color mode for reading the images. (-1 (default) => color, 0 => bg).')
@click.option('--output_img_ext', default=OUTPUT_IMG_EXTENSION, help='Extension for output images. Default => .jpg')
@click.option('--verbose', is_flag=True, help='Option to show images while reading them.')
@click.option('--false_data', is_flag=True, help='Option for adding false data from datasets parsers if available.')
def main(root_path, train_pct, test_pct, color_mode, verbose, false_data, output_img_ext):
    # Path of the training and testing txt used as input for darknet.
    if root_path[-1] != "/":
        root_path += "/"
    output_train_text_path = root_path + "train.txt"
    output_test_text_path = root_path + "test.txt"
    # Path of the resulting training and testing images of this script and labels.
    output_train_dir_path = root_path + "train/"
    output_test_dir_path = root_path + "test/"

    classes_counter_train_total = classes_counter_train.copy()
    classes_counter_test_total = classes_counter_test.copy()

    print("GTSDB DATASET: ")

    # Update the dataset variables
    update_global_variables(
        train_pct, test_pct, color_mode, verbose, false_data, output_img_ext
    )

    # Read dataset
    classes_counter_train_partial, classes_counter_test_partial = read_dataset(
        output_train_text_path,
        output_test_text_path,
        output_train_dir_path,
        output_test_dir_path,
    )
    classes_counter_train_total = add_arrays(
        classes_counter_train_total, classes_counter_train_partial
    )
    classes_counter_test_total = add_arrays(
        classes_counter_test_total, classes_counter_test_partial
    )

    print_db_info(classes_counter_train_partial, classes_counter_test_partial)

    print("TOTAL DATASET: ")
    print_db_info(classes_counter_train_total, classes_counter_test_total)


main()
