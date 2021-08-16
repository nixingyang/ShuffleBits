import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from absl import app, flags
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from reconstruction_utils import EvaluateAccuracies, get_feature_arrays

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from callbacks import HistoryLogger
from datasets import (
    _get_attribute_name_to_label_encoder_dict,
    load_accumulated_info_of_dataset,
)
from utils.vis_utils import summarize_model, visualize_model

flags.DEFINE_string("root_folder_path", "Dataset", "Folder path of the dataset.")
flags.DEFINE_string("dataset_name", "Market1501", "Name of the dataset.")
# ["Market1501", "DukeMTMC_reID"]
flags.DEFINE_string(
    "feature_file_path", "msmt_bot_R50_Market1501.npz", "File path of the feature."
)
flags.DEFINE_integer("block_num", 1, "Number of blocks for classification.")
flags.DEFINE_float("dropout_rate", 0.0, "Fraction of the input units to drop.")
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate.")
flags.DEFINE_integer("batch_size", 128, "Size of the batch.")
flags.DEFINE_integer("epoch_num", 100, "Number of epochs.")
flags.DEFINE_string(
    "output_folder_path",
    os.path.abspath(
        os.path.join(
            __file__, "../../output_{}".format(datetime.now().strftime("%Y_%m_%d"))
        )
    ),
    "Path to directory to output files.",
)
FLAGS = flags.FLAGS


def apply_post_processing(accumulated_info_dataframe, keywords=("down", "up")):
    print(
        "Before apply_post_processing: {} records".format(
            len(accumulated_info_dataframe)
        )
    )

    # Drop rows which contain missing values
    accumulated_info_dataframe = accumulated_info_dataframe.dropna(inplace=False)

    # Drop the "identity_ID" column
    accumulated_info_dataframe.drop(columns=["identity_ID"], inplace=True)

    # Iterate over keywords
    for keyword in keywords:
        # Find corresponding columns
        columns = [
            column
            for column in accumulated_info_dataframe.columns
            if column.startswith(keyword) and len(column) > len(keyword)
        ]
        print("Handling {} ...".format(" ".join(columns)))

        # Drop those columns
        attribute_array = accumulated_info_dataframe[columns].to_numpy()
        accumulated_info_dataframe.drop(columns=columns, inplace=True)
        assert len(np.unique(attribute_array)) == 2

        # Filter out problematic entries
        mask_array = np.sum(attribute_array == 2, axis=1) == 1
        accumulated_info_dataframe = accumulated_info_dataframe[mask_array]
        attribute_array = attribute_array[mask_array]

        # Add the new column
        accumulated_info_dataframe["{}_colour".format(keyword)] = np.where(
            attribute_array == 2
        )[1]

    print(
        "After apply_post_processing: {} records".format(
            len(accumulated_info_dataframe)
        )
    )

    return accumulated_info_dataframe


def get_groundtruth_arrays(
    accumulated_info_dataframe, attribute_name_to_label_encoder_dict
):
    groundtruth_arrays = []
    for attribute_name, label_encoder in attribute_name_to_label_encoder_dict.items():
        groundtruth_array = label_encoder.transform(
            accumulated_info_dataframe[attribute_name].to_numpy()
        )
        groundtruth_array = to_categorical(
            groundtruth_array, num_classes=len(label_encoder.classes_)
        )
        groundtruth_arrays.append(groundtruth_array)
    return groundtruth_arrays


def init_model(
    feature_shape, attribute_name_to_label_encoder_dict, block_num, dropout_rate
):
    # Define the model
    input_tensor = Input(shape=feature_shape)
    output_tensor_list = []
    for attribute_name, label_encoder in attribute_name_to_label_encoder_dict.items():
        # Calculate the factor
        factor = np.exp(
            np.log(len(label_encoder.classes_) / feature_shape[0]) / block_num
        )
        units_list = feature_shape[0] * np.array(
            [np.power(factor, item) for item in np.arange(block_num) + 1]
        )
        units_list[-1] = len(label_encoder.classes_)
        units_list = units_list.astype(np.int)

        # Add the blocks
        output_tensor = input_tensor
        for index, units in enumerate(units_list, start=1):
            is_last_block = index == len(units_list)
            activation = "softmax" if is_last_block else "relu"
            name = attribute_name if is_last_block else None
            output_tensor = BatchNormalization()(output_tensor)
            output_tensor = Dropout(rate=dropout_rate)(output_tensor)
            output_tensor = Dense(units=units, activation=activation, name=name)(
                output_tensor
            )
        output_tensor_list.append(output_tensor)
    training_model = Model(inputs=[input_tensor], outputs=output_tensor_list)

    # Compile the model
    categorical_crossentropy_loss_function = (
        lambda y_true, y_pred: 1.0
        * categorical_crossentropy(
            y_true, y_pred, from_logits=False, label_smoothing=0.0
        )
    )
    classification_loss_function_list = [categorical_crossentropy_loss_function] * len(
        training_model.outputs
    )
    training_model.compile(optimizer=Adam(), loss=classification_loss_function_list)

    # Print the summary
    summarize_model(training_model)

    return training_model


def get_class_weight(groundtruth_arrays, output_names):
    # https://github.com/keras-team/keras/issues/4735
    # https://github.com/tensorflow/tensorflow/issues/41448
    class_weight_list = []
    for groundtruth_array in groundtruth_arrays:
        classes = np.arange(groundtruth_array.shape[1])
        y = np.argmax(groundtruth_array, axis=1)
        class_weight = compute_class_weight(
            class_weight="balanced", classes=classes, y=y
        )
        class_weight = dict(zip(classes, class_weight))
        class_weight_list.append(class_weight)
    class_weight = dict(zip(output_names, class_weight_list))
    return class_weight


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    root_folder_path, dataset_name = FLAGS.root_folder_path, FLAGS.dataset_name
    feature_file_path = FLAGS.feature_file_path
    block_num, dropout_rate = FLAGS.block_num, FLAGS.dropout_rate
    learning_rate, batch_size, epoch_num = (
        FLAGS.learning_rate,
        FLAGS.batch_size,
        FLAGS.epoch_num,
    )
    output_folder_path = os.path.join(
        FLAGS.output_folder_path, os.path.basename(feature_file_path).split(".")[0]
    )

    print("Creating the output folder at {} ...".format(output_folder_path))
    os.makedirs(output_folder_path, exist_ok=True)

    print("Loading the annotations ...")
    (
        train_accumulated_info_dataframe,
        query_accumulated_info_dataframe,
        gallery_accumulated_info_dataframe,
        _,
    ) = load_accumulated_info_of_dataset(
        root_folder_path=root_folder_path, dataset_name=dataset_name
    )
    valid_accumulated_info_dataframe = pd.concat(
        [query_accumulated_info_dataframe, gallery_accumulated_info_dataframe]
    )
    train_accumulated_info_dataframe = apply_post_processing(
        train_accumulated_info_dataframe
    )
    valid_accumulated_info_dataframe = apply_post_processing(
        valid_accumulated_info_dataframe
    )
    attribute_name_to_label_encoder_dict = _get_attribute_name_to_label_encoder_dict(
        train_accumulated_info_dataframe
    )
    train_groundtruth_arrays = get_groundtruth_arrays(
        accumulated_info_dataframe=train_accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict=attribute_name_to_label_encoder_dict,
    )
    valid_groundtruth_arrays = get_groundtruth_arrays(
        accumulated_info_dataframe=valid_accumulated_info_dataframe,
        attribute_name_to_label_encoder_dict=attribute_name_to_label_encoder_dict,
    )

    train_feature_array, valid_feature_array = (
        get_feature_arrays(  # pylint: disable=unbalanced-tuple-unpacking
            feature_file_path=feature_file_path,
            accumulated_info_dataframe_list=[
                train_accumulated_info_dataframe,
                valid_accumulated_info_dataframe,
            ],
            root_folder_path=root_folder_path,
        )
    )

    print("Initiating the model ...")
    training_model = init_model(
        feature_shape=train_feature_array.shape[
            1:
        ],  # pylint: disable=unsubscriptable-object
        attribute_name_to_label_encoder_dict=attribute_name_to_label_encoder_dict,
        block_num=block_num,
        dropout_rate=dropout_rate,
    )
    visualize_model(model=training_model, output_folder_path=output_folder_path)

    print("Training for {} epochs.".format(epoch_num))
    schedule = lambda epoch: (
        learning_rate if epoch < epoch_num / 2 else learning_rate / 5
    )
    learningratescheduler_callback = LearningRateScheduler(schedule=schedule, verbose=1)
    evaluateaccuracies_callback = EvaluateAccuracies(
        feature_array=valid_feature_array,
        groundtruth_arrays=valid_groundtruth_arrays,
        batch_size=batch_size,
        prefix="valid",
    )
    optimal_model_file_path = os.path.join(output_folder_path, "training_model.h5")
    modelcheckpoint_callback = ModelCheckpoint(
        filepath=optimal_model_file_path,
        monitor="{}_{}".format(evaluateaccuracies_callback.prefix, "mean"),
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    historylogger_callback = HistoryLogger(
        output_folder_path=os.path.join(output_folder_path, "training")
    )
    training_model.fit(
        x=train_feature_array,
        y=train_groundtruth_arrays,
        class_weight=get_class_weight(
            groundtruth_arrays=train_groundtruth_arrays,
            output_names=training_model.output_names,
        ),
        batch_size=batch_size,
        callbacks=[
            learningratescheduler_callback,
            evaluateaccuracies_callback,
            modelcheckpoint_callback,
            historylogger_callback,
        ],
        epochs=epoch_num,
        verbose=2,
    )


if __name__ == "__main__":
    app.run(main)
