import glob
import os
import sys

import numpy as np
from absl import app, flags
from sklearn.metrics import pairwise_distances

flags.DEFINE_string(
    "source_feature_file_path",
    "msmt_bot_R50_Market1501.npz",
    "File path of the source feature.",
)
flags.DEFINE_string(
    "target_feature_folder_path",
    "Re-Identification_*",
    "Folder path of the target feature.",
)
FLAGS = flags.FLAGS


def load_feature_file(feature_file_path):
    with np.load(feature_file_path) as data:
        accumulated_image_file_paths = data["accumulated_image_file_paths"]
        accumulated_feature_array = data["accumulated_feature_array"]
    return accumulated_image_file_paths, accumulated_feature_array


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    source_feature_file_path = FLAGS.source_feature_file_path
    target_feature_folder_path = FLAGS.target_feature_folder_path

    print("Loading {} ...".format(source_feature_file_path))
    identifier = os.path.basename(source_feature_file_path).split(".")[0]
    source_image_file_paths, source_feature_array = load_feature_file(
        source_feature_file_path
    )
    image_file_path_to_feature_vector = dict(
        zip(source_image_file_paths, source_feature_array)
    )

    target_feature_file_paths = sorted(
        glob.glob(os.path.join(target_feature_folder_path, "**/*.npz"), recursive=True)
    )
    for target_feature_file_path in target_feature_file_paths:
        if identifier not in target_feature_file_path:
            continue

        print("Loading {} ...".format(target_feature_file_path))
        target_image_file_paths, target_feature_array = load_feature_file(
            target_feature_file_path
        )

        try:
            source_feature_array = np.vstack(
                [
                    image_file_path_to_feature_vector[image_file_path]
                    for image_file_path in target_image_file_paths
                ]
            )
            distances = []
            for source_feature_vector, target_feature_vector in zip(
                source_feature_array, target_feature_array
            ):
                distance = pairwise_distances(
                    [source_feature_vector], [target_feature_vector], metric="cosine"
                )[0, 0]
                distances.append(distance)
            print("There are {} samples in total.".format(len(distances)))
            print("Mean distance is {}.".format(np.mean(distances)))
        except Exception as exception:  # pylint: disable=broad-except
            print(exception)
            print("Failed to process {} ...".format(target_feature_file_path))
        finally:
            print("\n")


if __name__ == "__main__":
    app.run(main)
