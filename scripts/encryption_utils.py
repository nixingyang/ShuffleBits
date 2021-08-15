"""
References:
https://en.wikipedia.org/wiki/Single-precision_floating-point_format
"""

import numpy as np
import tensorflow as tf

int_to_binary_in_numpy = lambda x: np.binary_repr(x, width=32)
binary_to_int_in_numpy = lambda x: int(x, base=2)

# https://stackoverflow.com/a/49997506
int_to_binary_in_tensorflow = lambda x: tf.reverse(
    tf.math.mod(tf.bitwise.right_shift(tf.constant(x), tf.range(32)), 2), axis=[0]
)

# https://stackoverflow.com/a/42163515
binary_to_int_in_tensorflow = lambda x: tf.reduce_sum(
    tf.reverse(x, axis=[0]) * 2 ** tf.range(tf.size(x))
)

discard_nan = lambda x: x[np.logical_not(np.isnan(x))]


def get_index_array(seed=0):
    index_array = np.arange(32)
    random_generator = np.random.default_rng(seed=seed)
    random_generator.shuffle(index_array)
    return index_array


def apply_conversion_on_number_in_numpy(number_in_float32, index_array):
    number_in_int32 = number_in_float32.view(
        dtype=np.int32
    )  # pylint: disable=unexpected-keyword-arg
    number_in_binary32 = int_to_binary_in_numpy(number_in_int32)
    number_in_binary32 = "".join((map(number_in_binary32.__getitem__, index_array)))
    number_in_int32 = np.int32(binary_to_int_in_numpy(number_in_binary32))
    number_in_float32 = number_in_int32.view(dtype=np.float32)
    return number_in_float32


def apply_conversion_on_number_in_tensorflow(number_in_float32, index_array):
    number_in_int32 = tf.bitcast(number_in_float32, tf.int32)
    number_in_binary32 = int_to_binary_in_tensorflow(number_in_int32)
    number_in_binary32 = tf.gather(number_in_binary32, index_array)
    number_in_int32 = binary_to_int_in_tensorflow(number_in_binary32)
    number_in_float32 = tf.bitcast(number_in_int32, tf.float32).numpy()
    return number_in_float32


def apply_conversion_on_array(array_in_float32, index_array, backend="numpy"):
    apply_conversion_on_number = {
        "numpy": apply_conversion_on_number_in_numpy,
        "tensorflow": apply_conversion_on_number_in_tensorflow,
    }[backend]
    converted = np.array(
        [
            apply_conversion_on_number(number_in_float32, index_array)
            for number_in_float32 in np.ravel(array_in_float32)
        ]
    )
    converted = converted.reshape(array_in_float32.shape)
    return converted


def sanity_check(size=(100, 100)):
    index_array_for_encryption = get_index_array()
    index_array_for_decryption = np.argsort(index_array_for_encryption)
    original_array = np.random.uniform(low=-1.0, high=1.0, size=size).astype(np.float32)

    # Check the numpy backend
    encrypted_array_in_numpy = apply_conversion_on_array(
        original_array, index_array_for_encryption, backend="numpy"
    )
    decrypted_array_in_numpy = apply_conversion_on_array(
        encrypted_array_in_numpy, index_array_for_decryption, backend="numpy"
    )
    assert np.array_equal(original_array, decrypted_array_in_numpy)

    # Check the tensorflow backend
    encrypted_array_in_tensorflow = apply_conversion_on_array(
        original_array, index_array_for_encryption, backend="tensorflow"
    )
    decrypted_array_in_tensorflow = apply_conversion_on_array(
        encrypted_array_in_tensorflow, index_array_for_decryption, backend="tensorflow"
    )
    assert np.array_equal(original_array, decrypted_array_in_tensorflow)

    # Compare the results of numpy and tensorflow
    assert np.array_equal(
        discard_nan(encrypted_array_in_numpy),
        discard_nan(encrypted_array_in_tensorflow),
    )

    print("All done!")


if __name__ == "__main__":
    sanity_check()
