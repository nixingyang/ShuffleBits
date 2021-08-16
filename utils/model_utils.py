import os
import sys

import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.keras.models import Model, clone_model, model_from_json
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers.convolutional import Conv


def construct_intermediate_model(model, intermediate_layer_name):
    # https://github.com/tensorflow/tensorflow/issues/34977#issuecomment-571878943
    intermediate_layer = model.get_layer(intermediate_layer_name)
    node_index = 1 if isinstance(intermediate_layer, Model) else 0
    model = Model(
        inputs=model.input, outputs=intermediate_layer.get_output_at(node_index)
    )
    return model


def modify_input_shape(model, input_shape, custom_objects=None):
    if np.array_equal(model.input_shape[1:], input_shape):
        return model
    vanilla_weights = model.get_weights()
    model._layers[0]._batch_input_shape = (
        None,
        *input_shape,
    )  # pylint: disable=protected-access
    model = model_from_json(json_string=model.to_json(), custom_objects=custom_objects)
    model.set_weights(vanilla_weights)
    return model
