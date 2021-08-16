import os
import sys
from datetime import datetime

import pandas as pd
import tensorflow as tf
from absl import app, flags
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from reconstruction_utils import (
    DataSequence,
    GeneratePredictions,
    apply_decoder,
    flatten_loss_function,
    generate_feature_maps,
    get_feature_arrays,
    integrate_preprocess_input,
    save_predictions,
)

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from callbacks import HistoryLogger
from datasets import load_accumulated_info_of_dataset
from utils.vis_utils import summarize_model, visualize_model

flags.DEFINE_string("root_folder_path", "Dataset", "Folder path of the dataset.")
flags.DEFINE_string("dataset_name", "Market1501", "Name of the dataset.")
# ["Market1501", "DukeMTMC_reID"]
flags.DEFINE_string(
    "feature_file_path", "msmt_bot_R50_Market1501.npz", "File path of the feature."
)
flags.DEFINE_integer("image_width", 128, "Width of the images.")
flags.DEFINE_integer("image_height", 384, "Height of the images.")
flags.DEFINE_string(
    "feature_reconstruction_model_file_path",
    os.path.expanduser("~/.keras/models/Complete_ResNet50.h5"),
    "File path of the feature reconstruction model.",
)
flags.DEFINE_string(
    "feature_reconstruction_model_intermediate_layer_name",
    "conv2_block3_out",
    "Name of intermediate layer of the feature reconstruction model.",
)
# "Complete_ResNet50.h5": "conv2_block3_out"
# "Complete_VGG16.h5": "block2_pool"
flags.DEFINE_string(
    "feature_reconstruction_model_preprocess_input_mode",
    "caffe",
    "Mode of preprocess_input of the feature reconstruction model.",
)
flags.DEFINE_float("pixel_loss_weight", 1.0, "Weight of the pixel loss.")
flags.DEFINE_float(
    "feature_reconstruction_loss_weight",
    1.0,
    "Weight of the feature reconstruction loss.",
)
flags.DEFINE_float(
    "total_variation_loss_weight", 0.0, "Weight of the total variation loss."
)
flags.DEFINE_float("learning_rate_of_G", 0.0001, "Learning rate of generator.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch.")
flags.DEFINE_integer("epoch_num", 300, "Number of epochs.")
flags.DEFINE_integer(
    "train_steps_per_epoch", 200, "Number of steps per epoch for training."
)
flags.DEFINE_integer(
    "valid_steps_per_epoch", 50, "Number of steps per epoch for validation."
)
flags.DEFINE_integer("prediction_num", 4, "Number of predictions per epoch.")
flags.DEFINE_integer("workers", 5, "Number of processes to spin up for data generator.")
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


def init_model(
    image_shape,
    feature_shape,
    feature_reconstruction_model_file_path,
    feature_reconstruction_model_intermediate_layer_name,
    feature_reconstruction_model_preprocess_input_mode,
    pixel_loss_weight,
    feature_reconstruction_loss_weight,
    total_variation_loss_weight,
    learning_rate_of_G,
):
    """
    from tensorflow.keras.applications import ResNet50
    model = ResNet50(include_top=False, weights="imagenet")
    model.save(os.path.expanduser("~/.keras/models/Complete_ResNet50.h5"))
    """
    print("Initializing a feature reconstruction model ...")
    feature_reconstruction_model = integrate_preprocess_input(
        vanilla_model_file_path=feature_reconstruction_model_file_path,
        vanilla_model_intermediate_layer_name=feature_reconstruction_model_intermediate_layer_name,
        vanilla_model_preprocess_input_mode=feature_reconstruction_model_preprocess_input_mode,
        vanilla_model_input_shape=image_shape,
        name="feature_reconstruction",
    )
    feature_reconstruction_model.trainable = False

    print("Initializing a decoder ...")
    decoder_input_tensor = Input(shape=feature_shape)
    decoder_output_tensor, factor, num = generate_feature_maps(
        input_tensor=decoder_input_tensor, input_shape=image_shape
    )
    decoder_output_tensor = apply_decoder(
        input_tensor=decoder_output_tensor, factor=factor, num=num
    )
    decoder_model = Model(
        inputs=[decoder_input_tensor], outputs=[decoder_output_tensor], name="decoder"
    )

    print("Initializing a model for training ...")
    ground_truth_image_tensor = Input(shape=image_shape)
    encoded_tensor = Input(shape=feature_shape)
    ground_truth_image_feature_tensor = feature_reconstruction_model(
        Lambda(lambda x: x * 255)(ground_truth_image_tensor)
    )
    reconstructed_image_tensor = decoder_model(encoded_tensor)
    reconstructed_image_feature_tensor = feature_reconstruction_model(
        Lambda(lambda x: x * 255)(reconstructed_image_tensor)
    )
    training_model = Model(
        inputs=[ground_truth_image_tensor, encoded_tensor],
        outputs=[reconstructed_image_tensor],
    )
    feature_reconstruction_loss = K.mean(
        flatten_loss_function()(
            ground_truth_image_feature_tensor, reconstructed_image_feature_tensor
        )
    )
    training_model.add_metric(
        feature_reconstruction_loss, name="feature_reconstruction", aggregation="mean"
    )
    if feature_reconstruction_loss_weight > 0:
        feature_reconstruction_loss *= feature_reconstruction_loss_weight
        training_model.add_loss(feature_reconstruction_loss)
    total_variation_loss = K.mean(
        tf.image.total_variation(reconstructed_image_tensor)
        / tf.dtypes.cast(
            K.prod(K.int_shape(reconstructed_image_tensor)[1:]), tf.float32
        )
    )
    training_model.add_metric(
        total_variation_loss, name="total_variation", aggregation="mean"
    )
    if total_variation_loss_weight > 0:
        total_variation_loss *= total_variation_loss_weight
        training_model.add_loss(total_variation_loss)
    training_model.compile(
        optimizer=Adam(learning_rate=learning_rate_of_G),
        loss=flatten_loss_function(),
        loss_weights=[pixel_loss_weight],
    )

    # Print the summary
    summarize_model(training_model)

    return training_model, decoder_model


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    root_folder_path, dataset_name = FLAGS.root_folder_path, FLAGS.dataset_name
    feature_file_path = FLAGS.feature_file_path
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    image_shape = (image_height, image_width, 3)
    (
        feature_reconstruction_model_file_path,
        feature_reconstruction_model_intermediate_layer_name,
        feature_reconstruction_model_preprocess_input_mode,
    ) = (
        FLAGS.feature_reconstruction_model_file_path,
        FLAGS.feature_reconstruction_model_intermediate_layer_name,
        FLAGS.feature_reconstruction_model_preprocess_input_mode,
    )
    (
        pixel_loss_weight,
        feature_reconstruction_loss_weight,
        total_variation_loss_weight,
    ) = (
        FLAGS.pixel_loss_weight,
        FLAGS.feature_reconstruction_loss_weight,
        FLAGS.total_variation_loss_weight,
    )
    learning_rate_of_G, batch_size, epoch_num = (
        FLAGS.learning_rate_of_G,
        FLAGS.batch_size,
        FLAGS.epoch_num,
    )
    train_steps_per_epoch, valid_steps_per_epoch, prediction_num = (
        FLAGS.train_steps_per_epoch,
        FLAGS.valid_steps_per_epoch,
        FLAGS.prediction_num,
    )
    workers = FLAGS.workers
    use_multiprocessing = workers > 1
    output_folder_path = os.path.join(
        FLAGS.output_folder_path,
        os.path.basename(feature_file_path).split(".")[0],
        "{}_{}_{}".format(
            pixel_loss_weight,
            feature_reconstruction_loss_weight,
            total_variation_loss_weight,
        ),
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
    training_model, decoder_model = init_model(
        image_shape=image_shape,
        feature_shape=train_feature_array.shape[
            1:
        ],  # pylint: disable=unsubscriptable-object
        feature_reconstruction_model_file_path=feature_reconstruction_model_file_path,
        feature_reconstruction_model_intermediate_layer_name=feature_reconstruction_model_intermediate_layer_name,
        feature_reconstruction_model_preprocess_input_mode=feature_reconstruction_model_preprocess_input_mode,
        pixel_loss_weight=pixel_loss_weight,
        feature_reconstruction_loss_weight=feature_reconstruction_loss_weight,
        total_variation_loss_weight=total_variation_loss_weight,
        learning_rate_of_G=learning_rate_of_G,
    )
    visualize_model(model=training_model, output_folder_path=output_folder_path)

    train_generator = DataSequence(
        image_file_path_array=train_accumulated_info_dataframe[
            "image_file_path"
        ].values,
        feature_array=train_feature_array,
        input_shape=image_shape,
        batch_size=batch_size,
        steps_per_epoch=train_steps_per_epoch,
    )
    valid_generator = DataSequence(
        image_file_path_array=valid_accumulated_info_dataframe[
            "image_file_path"
        ].values,
        feature_array=valid_feature_array,
        input_shape=image_shape,
        batch_size=batch_size,
        steps_per_epoch=valid_steps_per_epoch,
    )
    train_prediction_generator = DataSequence(
        image_file_path_array=train_accumulated_info_dataframe[
            "image_file_path"
        ].values,
        feature_array=train_feature_array,
        input_shape=image_shape,
        batch_size=prediction_num,
        steps_per_epoch=1,
        seed=0,
    )
    valid_prediction_generator = DataSequence(
        image_file_path_array=valid_accumulated_info_dataframe[
            "image_file_path"
        ].values,
        feature_array=valid_feature_array,
        input_shape=image_shape,
        batch_size=prediction_num,
        steps_per_epoch=1,
        seed=0,
    )
    train_prediction_callback = GeneratePredictions(
        prediction_generator=train_prediction_generator,
        output_folder_path=os.path.join(output_folder_path, "train_prediction"),
    )
    valid_prediction_callback = GeneratePredictions(
        prediction_generator=valid_prediction_generator,
        output_folder_path=os.path.join(output_folder_path, "valid_prediction"),
    )
    training_model_file_path = os.path.join(output_folder_path, "training_model.h5")
    modelcheckpoint_callback = ModelCheckpoint(
        filepath=training_model_file_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    historylogger_callback = HistoryLogger(
        output_folder_path=os.path.join(output_folder_path, "training")
    )
    if os.path.isfile(training_model_file_path):
        print("Skipping training ...")
        training_model.load_weights(training_model_file_path)
        save_predictions(
            model=decoder_model,
            batch_size=batch_size,
            image_file_path_array=valid_accumulated_info_dataframe[
                "image_file_path"
            ].values,
            root_folder_path=root_folder_path,
            feature_array=valid_feature_array,
            output_folder_path=os.path.join(output_folder_path, "complete_valid"),
        )
    else:
        print("Training for {} epochs.".format(epoch_num))
        training_model.fit(
            x=train_generator,
            validation_data=valid_generator,
            callbacks=[
                train_prediction_callback,
                valid_prediction_callback,
                modelcheckpoint_callback,
                historylogger_callback,
            ],
            epochs=epoch_num,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            verbose=2,
        )


if __name__ == "__main__":
    app.run(main)
