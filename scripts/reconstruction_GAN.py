import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from reconstruction_utils import (
    DataSequence,
    GeneratePredictions,
    apply_decoder,
    apply_discriminator,
    flatten_loss_function,
    generate_feature_maps,
    get_feature_arrays,
    integrate_preprocess_input,
    save_predictions,
)

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
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
flags.DEFINE_bool(
    "use_spectral_normalization_in_discriminator",
    False,
    "Use spectral normalization in discriminator.",
)
flags.DEFINE_string(
    "comparison_mode_in_discriminator",
    "concatenation",
    "Comparison mode in discriminator.",
)
# ["concatenation", "absolute_difference"]
flags.DEFINE_float("pixel_loss_weight", 1.0, "Weight of the pixel loss.")
flags.DEFINE_float(
    "feature_reconstruction_loss_weight",
    1.0,
    "Weight of the feature reconstruction loss.",
)
flags.DEFINE_float(
    "total_variation_loss_weight", 0.0, "Weight of the total variation loss."
)
flags.DEFINE_float(
    "discriminator_loss_weight", 1.0, "Weight of the loss in discriminator."
)
flags.DEFINE_float("learning_rate_of_G", 0.0001, "Learning rate of generator.")
flags.DEFINE_float("learning_rate_of_D", 0.0001, "Learning rate of discriminator.")
flags.DEFINE_integer("batch_size", 64, "Size of the batch.")
flags.DEFINE_integer("epoch_num", 300, "Number of epochs.")
flags.DEFINE_integer(
    "train_steps_per_epoch", 200, "Number of steps per epoch for training."
)
flags.DEFINE_integer(
    "valid_steps_per_epoch", 50, "Number of steps per epoch for validation."
)
flags.DEFINE_integer("prediction_num", 4, "Number of predictions per epoch.")
flags.DEFINE_integer(
    "discriminator_steps", 1, "Number of consecutive steps to train discriminator."
)
flags.DEFINE_integer(
    "generator_and_discriminator_steps",
    1,
    "Number of consecutive steps to train generator.",
)
flags.DEFINE_bool(
    "use_different_batches",
    True,
    "Use different batches for discriminator and generator.",
)
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
    use_spectral_normalization_in_discriminator,
    comparison_mode_in_discriminator,
    pixel_loss_weight,
    feature_reconstruction_loss_weight,
    total_variation_loss_weight,
    discriminator_loss_weight,
    learning_rate_of_G,
    learning_rate_of_D,
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

    print("Initializing a discriminator ...")
    discriminator_input_target_tensor = Input(shape=image_shape)
    discriminator_input_condition_tensor = Input(shape=feature_shape)
    discriminator_output_tensor = apply_discriminator(
        input_target_tensor=discriminator_input_target_tensor,
        input_condition_tensor=discriminator_input_condition_tensor,
        factor=factor,
        num=num,
        use_spectral_normalization=use_spectral_normalization_in_discriminator,
        comparison_mode=comparison_mode_in_discriminator,
    )
    discriminator_model = Model(
        inputs=[
            discriminator_input_target_tensor,
            discriminator_input_condition_tensor,
        ],
        outputs=[discriminator_output_tensor],
        name="discriminator",
    )
    discriminator_model.compile(
        optimizer=Adam(learning_rate=learning_rate_of_D), loss="mean_squared_error"
    )

    print("Initializing a merged model using generator and discriminator ...")
    ground_truth_image_tensor = Input(shape=image_shape)
    encoded_tensor = Input(shape=feature_shape)
    ground_truth_image_feature_tensor = feature_reconstruction_model(
        Lambda(lambda x: x * 255)(ground_truth_image_tensor)
    )
    reconstructed_image_tensor = decoder_model(encoded_tensor)
    reconstructed_image_feature_tensor = feature_reconstruction_model(
        Lambda(lambda x: x * 255)(reconstructed_image_tensor)
    )
    visualization_model = Model(
        inputs=[ground_truth_image_tensor, encoded_tensor],
        outputs=[reconstructed_image_tensor],
    )
    discriminator_model.trainable = False
    probability_tensor = discriminator_model(
        [reconstructed_image_tensor, encoded_tensor]
    )
    generator_and_discriminator_model = Model(
        inputs=[ground_truth_image_tensor, encoded_tensor],
        outputs=[reconstructed_image_tensor, probability_tensor],
    )
    feature_reconstruction_loss = K.mean(
        flatten_loss_function()(
            ground_truth_image_feature_tensor, reconstructed_image_feature_tensor
        )
    )
    generator_and_discriminator_model.add_metric(
        feature_reconstruction_loss, name="feature_reconstruction", aggregation="mean"
    )
    if feature_reconstruction_loss_weight > 0:
        feature_reconstruction_loss *= feature_reconstruction_loss_weight
        generator_and_discriminator_model.add_loss(feature_reconstruction_loss)
    total_variation_loss = K.mean(
        tf.image.total_variation(reconstructed_image_tensor)
        / tf.dtypes.cast(
            K.prod(K.int_shape(reconstructed_image_tensor)[1:]), tf.float32
        )
    )
    generator_and_discriminator_model.add_metric(
        total_variation_loss, name="total_variation", aggregation="mean"
    )
    if total_variation_loss_weight > 0:
        total_variation_loss *= total_variation_loss_weight
        generator_and_discriminator_model.add_loss(total_variation_loss)
    generator_and_discriminator_model.compile(
        optimizer=Adam(learning_rate=learning_rate_of_G),
        loss=[flatten_loss_function(), "mean_squared_error"],
        loss_weights=[pixel_loss_weight, discriminator_loss_weight],
    )

    # Print the summary
    summarize_model(generator_and_discriminator_model)

    return (
        generator_and_discriminator_model,
        discriminator_model,
        visualization_model,
        decoder_model,
    )


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
    use_spectral_normalization_in_discriminator, comparison_mode_in_discriminator = (
        FLAGS.use_spectral_normalization_in_discriminator,
        FLAGS.comparison_mode_in_discriminator,
    )
    (
        pixel_loss_weight,
        feature_reconstruction_loss_weight,
        total_variation_loss_weight,
        discriminator_loss_weight,
    ) = (
        FLAGS.pixel_loss_weight,
        FLAGS.feature_reconstruction_loss_weight,
        FLAGS.total_variation_loss_weight,
        FLAGS.discriminator_loss_weight,
    )
    learning_rate_of_G, learning_rate_of_D, batch_size, epoch_num = (
        FLAGS.learning_rate_of_G,
        FLAGS.learning_rate_of_D,
        FLAGS.batch_size,
        FLAGS.epoch_num,
    )
    train_steps_per_epoch, valid_steps_per_epoch, prediction_num = (
        FLAGS.train_steps_per_epoch,
        FLAGS.valid_steps_per_epoch,
        FLAGS.prediction_num,
    )
    discriminator_steps, generator_and_discriminator_steps, use_different_batches = (
        FLAGS.discriminator_steps,
        FLAGS.generator_and_discriminator_steps,
        FLAGS.use_different_batches,
    )
    total_steps = discriminator_steps + generator_and_discriminator_steps
    output_folder_path = os.path.join(
        FLAGS.output_folder_path,
        os.path.basename(feature_file_path).split(".")[0],
        "{}_{}_{}_{}".format(
            pixel_loss_weight,
            feature_reconstruction_loss_weight,
            total_variation_loss_weight,
            discriminator_loss_weight,
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
    (
        generator_and_discriminator_model,
        discriminator_model,
        visualization_model,
        decoder_model,
    ) = init_model(
        image_shape=image_shape,
        feature_shape=train_feature_array.shape[
            1:
        ],  # pylint: disable=unsubscriptable-object
        feature_reconstruction_model_file_path=feature_reconstruction_model_file_path,
        feature_reconstruction_model_intermediate_layer_name=feature_reconstruction_model_intermediate_layer_name,
        feature_reconstruction_model_preprocess_input_mode=feature_reconstruction_model_preprocess_input_mode,
        use_spectral_normalization_in_discriminator=use_spectral_normalization_in_discriminator,
        comparison_mode_in_discriminator=comparison_mode_in_discriminator,
        pixel_loss_weight=pixel_loss_weight,
        feature_reconstruction_loss_weight=feature_reconstruction_loss_weight,
        total_variation_loss_weight=total_variation_loss_weight,
        discriminator_loss_weight=discriminator_loss_weight,
        learning_rate_of_G=learning_rate_of_G,
        learning_rate_of_D=learning_rate_of_D,
    )
    visualize_model(
        model=generator_and_discriminator_model, output_folder_path=output_folder_path
    )

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
    train_prediction_callback.model = visualization_model
    valid_prediction_callback.model = visualization_model
    generator_and_discriminator_model_file_path = os.path.join(
        output_folder_path, "generator_and_discriminator_model.h5"
    )
    if os.path.isfile(generator_and_discriminator_model_file_path):
        print("Skipping training ...")
        generator_and_discriminator_model.load_weights(
            generator_and_discriminator_model_file_path
        )
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
        for epoch_index in range(epoch_num):
            print("Epoch {}:".format(epoch_index))

            for batch_index in range(len(train_generator)):
                # Retrieve data
                input_array, groundtruth_array = train_generator[batch_index]
                ones_array = np.ones((len(groundtruth_array), 1), dtype=np.float32)
                zeros_array = np.zeros_like(ones_array)

                optimize_discriminator, optimize_generator = True, True
                if use_different_batches:
                    optimize_discriminator = np.remainder(
                        batch_index, total_steps
                    ) in range(discriminator_steps)
                    optimize_generator = not optimize_discriminator
                if discriminator_loss_weight == 0:
                    optimize_discriminator = False
                discriminator_loss, generator_and_discriminator_loss = None, None

                if optimize_discriminator:
                    # Get the reconstructed images
                    condition_array = input_array[1]
                    reconstructed_target_array = visualization_model.predict_on_batch(
                        input_array
                    )

                    # Optimize the discriminator
                    discriminator_loss = discriminator_model.train_on_batch(
                        x=[
                            np.vstack((groundtruth_array, reconstructed_target_array)),
                            np.vstack((condition_array, condition_array)),
                        ],
                        y=np.vstack((ones_array, zeros_array)),
                    )

                if optimize_generator:
                    # Optimize the generator
                    generator_and_discriminator_loss = (
                        generator_and_discriminator_model.train_on_batch(
                            x=input_array, y=[groundtruth_array, ones_array]
                        )
                    )

                # Print losses
                print(
                    "Batch {}: D {}, G&D {}".format(
                        batch_index,
                        discriminator_loss,
                        generator_and_discriminator_loss,
                    )
                )

            # Generate predictions
            train_prediction_callback.on_epoch_end(epoch=epoch_index)
            valid_prediction_callback.on_epoch_end(epoch=epoch_index)

            # Shuffle the data
            train_generator.on_epoch_end()
            valid_generator.on_epoch_end()

        print("Saving the model ...")
        generator_and_discriminator_model.save(
            generator_and_discriminator_model_file_path
        )


if __name__ == "__main__":
    app.run(main)
