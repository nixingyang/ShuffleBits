import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _load_accumulated_info(
    root_folder_path,
    dataset_folder_name="DukeMTMC-reID",
    image_folder_name="bounding_box_train",
):
    """
    References:
    https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
    gdrive download 1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
    7za x DukeMTMC-reID.zip
    sha256sum DukeMTMC-reID.zip
    932ae18937b6a77bc59846d4fb00da4ee02cdda93329ca0537ad899a569e3505  DukeMTMC-reID.zip
    """
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)
    image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    image_file_path_list = sorted(glob.glob(os.path.join(image_folder_path, "*.jpg")))
    if image_folder_name == "bounding_box_train":
        assert len(image_file_path_list) == 16522
    elif image_folder_name == "bounding_box_test":
        assert len(image_file_path_list) == 17661
    elif image_folder_name == "query":
        assert len(image_file_path_list) == 2228
    else:
        assert False, "{} is an invalid argument!".format(image_folder_name)

    # Improving Person Re-identification by Attribute and Identity Learning
    # https://github.com/vana77/DukeMTMC-attribute
    attribute_file_path = os.path.join(
        dataset_folder_path, "DukeMTMC-attribute", "duke_attribute.mat"
    )
    attribute_file_content = loadmat(attribute_file_path)["duke_attribute"][0, 0]
    train_attribute_file_content, test_attribute_file_content = (
        attribute_file_content["train"],
        attribute_file_content["test"],
    )
    assert sorted(train_attribute_file_content.dtype.names) == sorted(
        test_attribute_file_content.dtype.names
    )
    attribute_name_list = sorted(train_attribute_file_content.dtype.names)
    attribute_name_list.remove("image_index")
    identity_IDs, attribute_values = [], []
    for split_attribute_file_content in (
        train_attribute_file_content,
        test_attribute_file_content,
    ):
        identity_IDs.append(
            split_attribute_file_content["image_index"][0, 0].flatten().astype(np.int)
        )
        attribute_values.append(
            np.swapaxes(
                np.vstack(
                    [
                        split_attribute_file_content[attribute_name][0, 0].flatten()
                        for attribute_name in attribute_name_list
                    ]
                ),
                0,
                1,
            )
        )
    identity_IDs, attribute_values = np.hstack(identity_IDs).tolist(), np.vstack(
        attribute_values
    )

    accumulated_info_list = []
    for image_file_path in image_file_path_list:
        image_file_name = image_file_path.split(os.sep)[-1]
        identity_ID = int(image_file_name.split("_")[0])
        camera_ID = int(image_file_name.split("_")[1][1])
        # Append the records
        accumulated_info = {
            "image_file_path": image_file_path,
            "identity_ID": identity_ID,
            "camera_ID": camera_ID,
        }
        try:
            attribute_index = identity_IDs.index(identity_ID)
            for attribute_name, attribute_value in zip(
                attribute_name_list, attribute_values[attribute_index]
            ):
                accumulated_info[attribute_name] = attribute_value
        except ValueError:
            pass
        finally:
            accumulated_info_list.append(accumulated_info)

    # Convert list to data frame
    accumulated_info_dataframe = pd.DataFrame(accumulated_info_list)
    return accumulated_info_dataframe


def load_DukeMTMC_reID(root_folder_path):
    train_and_valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="bounding_box_train"
    )
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="bounding_box_test"
    )
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="query"
    )
    return (
        train_and_valid_accumulated_info_dataframe,
        test_query_accumulated_info_dataframe,
        test_gallery_accumulated_info_dataframe,
    )
