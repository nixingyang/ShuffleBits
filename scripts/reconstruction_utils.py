import os
import sys

import cv2
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from tensorflow_addons.layers import SpectralNormalization

sys.path.append(os.path.abspath(os.path.join(__file__, "../..")))
from utils.model_utils import construct_intermediate_model, modify_input_shape

deprocess_function = lambda x: cv2.cvtColor(
    (x * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
)


def get_feature_arrays(
    feature_file_path, accumulated_info_dataframe_list, root_folder_path
):

    def _load_feature_file():
        print("Loading the feature file {} ...".format(feature_file_path))
        with np.load(feature_file_path) as data:
            accumulated_image_file_paths = data["accumulated_image_file_paths"]
            accumulated_feature_array = data["accumulated_feature_array"]
        image_file_path_to_feature_vector_dict = dict(
            zip(accumulated_image_file_paths, accumulated_feature_array)
        )
        return image_file_path_to_feature_vector_dict

    def _map_feature_array(
        accumulated_info_dataframe, image_file_path_to_feature_vector_dict
    ):
        image_file_paths = [
            os.path.relpath(item, root_folder_path)
            for item in accumulated_info_dataframe["image_file_path"]
        ]
        feature_array = np.array(
            [
                image_file_path_to_feature_vector_dict[image_file_path]
                for image_file_path in image_file_paths
            ]
        )
        return feature_array

    feature_array_list = []
    image_file_path_to_feature_vector_dict = _load_feature_file()
    for accumulated_info_dataframe in accumulated_info_dataframe_list:
        feature_array = _map_feature_array(
            accumulated_info_dataframe=accumulated_info_dataframe,
            image_file_path_to_feature_vector_dict=image_file_path_to_feature_vector_dict,
        )
        feature_array_list.append(feature_array)
    return feature_array_list


def integrate_preprocess_input(  # pylint: disable=dangerous-default-value
    vanilla_model_file_path,
    vanilla_model_intermediate_layer_name,
    vanilla_model_preprocess_input_mode,
    vanilla_model_input_shape=None,
    append_pooling=False,
    name=None,
    custom_objects={},
):
    vanilla_model = load_model(vanilla_model_file_path, custom_objects=custom_objects)
    if len(vanilla_model_intermediate_layer_name) > 0:
        vanilla_model = construct_intermediate_model(
            vanilla_model, vanilla_model_intermediate_layer_name
        )
    if vanilla_model_input_shape is not None:
        vanilla_model = modify_input_shape(
            model=vanilla_model,
            input_shape=vanilla_model_input_shape,
            custom_objects=custom_objects,
        )
    input_tensor = Input(shape=vanilla_model.input_shape[1:])
    output_tensor = Lambda(
        lambda x: preprocess_input(x, mode=vanilla_model_preprocess_input_mode)
    )(input_tensor)
    output_tensor = vanilla_model(output_tensor)
    if append_pooling:
        output_tensor = GlobalAveragePooling2D()(output_tensor)
    integrated_model = Model(inputs=[input_tensor], outputs=[output_tensor], name=name)
    return integrated_model


def flatten_loss_function(vanilla_loss_function=mean_squared_error):
    return lambda y_true, y_pred: vanilla_loss_function(
        K.batch_flatten(y_true), K.batch_flatten(y_pred)
    )


def generate_feature_maps(
    input_tensor,
    input_shape,
    factor=2,
    dense_units=256,
    smallest_feature_maps_shape=(12, 4, 128),
):
    height_factor, width_factor = [
        input_shape[index] / smallest_feature_maps_shape[index] for index in range(2)
    ]
    assert height_factor == width_factor
    num = int(np.log(height_factor) / np.log(factor))
    assert np.power(factor, num) == height_factor
    output_tensor = input_tensor
    output_tensor = Dense(units=dense_units)(output_tensor)
    output_tensor = Activation("relu")(output_tensor)
    output_tensor = Dense(units=np.product(smallest_feature_maps_shape))(output_tensor)
    output_tensor = Activation("relu")(output_tensor)
    output_tensor = Reshape(smallest_feature_maps_shape)(output_tensor)
    return output_tensor, factor, num


def apply_decoder(input_tensor, factor, num, base_filters=16):

    def _add_upsampling(x, factor, filters):
        x1 = BatchNormalization()(x)
        x1 = Activation("relu")(x1)
        x1 = UpSampling2D(size=factor)(x1)
        x1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)
        x1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")(x1)
        x2 = UpSampling2D(size=factor)(x)
        x2 = Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x2)
        x3 = Add()([x1, x2])
        return x3

    output_tensor = input_tensor
    for index in range(num):
        output_tensor = _add_upsampling(
            x=output_tensor,
            factor=factor,
            filters=base_filters * factor ** (num - index - 1),
        )
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Activation("relu")(output_tensor)
    output_tensor = Conv2D(filters=3, kernel_size=3, strides=1, padding="same")(
        output_tensor
    )
    output_tensor = Lambda(lambda x: (K.tanh(x) + 1) / 2)(output_tensor)
    return output_tensor


def apply_discriminator(
    input_target_tensor,
    input_condition_tensor,
    factor,
    num,
    base_filters=32,
    use_spectral_normalization=True,
    comparison_mode="concatenation",
):

    def _add_downsampling(x, factor, filters, use_spectral_normalization):
        wrapper = SpectralNormalization if use_spectral_normalization else lambda x: x
        x1 = Activation("relu")(x)
        x1 = wrapper(Conv2D(filters=filters, kernel_size=3, strides=1, padding="same"))(
            x1
        )
        x1 = Activation("relu")(x1)
        x1 = wrapper(Conv2D(filters=filters, kernel_size=3, strides=1, padding="same"))(
            x1
        )
        x1 = AveragePooling2D(pool_size=factor)(x1)
        x2 = wrapper(Conv2D(filters=filters, kernel_size=1, strides=1, padding="same"))(
            x
        )
        x2 = AveragePooling2D(pool_size=factor)(x2)
        x3 = Add()([x1, x2])
        return x3

    output_target_tensor = input_target_tensor
    for index in range(num):
        output_target_tensor = _add_downsampling(
            x=output_target_tensor,
            factor=factor,
            filters=base_filters * factor**index,
            use_spectral_normalization=use_spectral_normalization,
        )
    output_target_tensor = GlobalAveragePooling2D()(output_target_tensor)
    output_condition_tensor = Dense(units=output_target_tensor.shape[1])(
        input_condition_tensor
    )
    if comparison_mode == "concatenation":
        output_tensor = Concatenate(axis=1)(
            [output_target_tensor, output_condition_tensor]
        )
    elif comparison_mode == "absolute_difference":
        output_tensor = Lambda(lambda x: K.abs(x[0] - x[1]))(
            [output_target_tensor, output_condition_tensor]
        )
    else:
        assert False, "{} is an invalid argument!".format(comparison_mode)
    output_tensor = Dense(units=1)(output_tensor)
    return output_tensor


class DataSequence(Sequence):

    def __init__(
        self,
        image_file_path_array,
        feature_array,
        input_shape,
        batch_size,
        steps_per_epoch,
        seed=None,
    ):
        super(DataSequence, self).__init__()

        # Save as variables
        self.image_file_path_array, self.feature_array = (
            image_file_path_array,
            feature_array,
        )
        self.input_shape = input_shape
        self.batch_size, self.steps_per_epoch = batch_size, steps_per_epoch
        self.iterations_per_epoch = batch_size * steps_per_epoch

        # Initiation
        self.indexes_generator = self._get_indexes_generator(
            num=len(image_file_path_array), seed=seed
        )
        self.indexes = next(self.indexes_generator)

    def _get_indexes_generator(self, num, seed):
        index_array = np.arange(num)
        random_generator = np.random.default_rng(seed=seed)

        index_list = []
        while True:
            random_generator.shuffle(index_array)
            for index in index_array:
                index_list.append(index)
                if len(index_list) == self.iterations_per_epoch:
                    yield np.array(index_list)
                    index_list = []

    def __len__(self):
        return self.steps_per_epoch

    def _read_image_file(self, image_file_path):
        # Read image file
        image_content = cv2.imread(image_file_path)

        # Resize the image
        image_content = cv2.resize(image_content, self.input_shape[:2][::-1])

        # Convert from BGR to RGB
        image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

        return image_content

    def __getitem__(self, index):
        # Get indexes in current batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        image_file_path_array = self.image_file_path_array[indexes]
        feature_array = self.feature_array[indexes]

        # Read images
        image_content_list = []
        for image_file_path in image_file_path_array:
            image_content = self._read_image_file(image_file_path)
            image_content_list.append(image_content)
        image_content_array = np.array(image_content_list, dtype=np.float32) / 255

        return (image_content_array, feature_array), image_content_array

    def on_epoch_end(self):
        self.indexes = next(self.indexes_generator)


class EvaluateAccuracies(Callback):

    def __init__(self, feature_array, groundtruth_arrays, batch_size=None, prefix=""):
        super(EvaluateAccuracies, self).__init__()

        self.feature_array = feature_array
        self.groundtruth_arrays = groundtruth_arrays
        self.batch_size = batch_size
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        # Generate predictions
        estimated_arrays = self.model.predict(
            self.feature_array, batch_size=self.batch_size
        )

        # Calculate scores
        score_list = []
        for groundtruth_array, estimated_array, output_name in zip(
            self.groundtruth_arrays, estimated_arrays, self.model.output_names
        ):
            score = balanced_accuracy_score(
                y_true=np.argmax(groundtruth_array, axis=1),
                y_pred=np.argmax(estimated_array, axis=1),
                adjusted=False,
            )
            score_list.append(score)
            logs["{}_{}".format(self.prefix, output_name)] = score

        # Get the mean score
        logs["{}_{}".format(self.prefix, "mean")] = np.mean(score_list)


class GeneratePredictions(Callback):

    def __init__(self, prediction_generator, output_folder_path):
        super(GeneratePredictions, self).__init__()

        self.prediction_generator = prediction_generator
        self.output_folder_path = output_folder_path
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)

    def on_epoch_end(self, epoch, logs=None):
        self.prediction_generator.on_epoch_end()
        input_array, groundtruth_array = self.prediction_generator[0]
        prediction_array = self.model.predict_on_batch(input_array)
        for index, (groundtruth, prediction) in enumerate(
            zip(groundtruth_array, prediction_array)
        ):
            cv2.imwrite(
                os.path.join(
                    self.output_folder_path,
                    "epoch_{}_index_{}_groundtruth.png".format(epoch, index),
                ),
                deprocess_function(groundtruth),
            )
            cv2.imwrite(
                os.path.join(
                    self.output_folder_path,
                    "epoch_{}_index_{}_prediction.png".format(epoch, index),
                ),
                deprocess_function(prediction),
            )


def save_predictions(
    model,
    batch_size,
    image_file_path_array,
    root_folder_path,
    feature_array,
    output_folder_path,
):
    sample_num = len(image_file_path_array)
    for indexes in np.array_split(np.arange(sample_num), sample_num // batch_size):
        image_file_paths = [
            os.path.join(output_folder_path, os.path.relpath(item, root_folder_path))
            for item in image_file_path_array[indexes]
        ]
        input_array = feature_array[indexes]
        prediction_array = model.predict_on_batch(input_array)
        for image_file_path, prediction in zip(image_file_paths, prediction_array):
            image_folder_path = os.path.abspath(os.path.join(image_file_path, ".."))
            os.makedirs(image_folder_path, exist_ok=True)
            cv2.imwrite(image_file_path, deprocess_function(prediction))
