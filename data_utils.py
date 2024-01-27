# Bronte Sihan Li, 2023

import re
from typing import Dict, List, Text, Tuple
import tensorflow as tf
import numpy as np

INPUT_FEATURES = [
    'elevation',
    'th',
    'vs',
    'tmmn',
    'tmmx',
    'sph',
    'pr',
    'pdsi',
    'NDVI',
    'population',
    'erc',
    'PrevFireMask',
]

OUTPUT_FEATURES = [
    'FireMask',
]

# Data statistics
# For each variable, the statistics are ordered in the form:
# (min_clip, max_clip, mean, standard deviation)
DATA_STATS = {
    # Elevation in m.
    # 0.1 percentile, 99.9 percentile
    'elevation': (0.0, 3141.0, 657.3003, 649.0147),
    # Pressure
    # 0.1 percentile, 99.9 percentile
    'pdsi': (-6.12974870967865, 7.876040384292651, -0.0052714925, 2.6823447),
    'NDVI': (-9821.0, 9996.0, 5157.625, 2466.6677),  # min, max
    # Precipitation in mm.
    # Negative values do not make sense, so min is set to 0.
    # 0., 99.9 percentile
    'pr': (0.0, 44.53038024902344, 1.7398051, 4.482833),
    # Specific humidity.
    # Negative values do not make sense, so min is set to 0.
    # The range of specific humidity is up to 100% so max is 1.
    'sph': (0.0, 1.0, 0.0071658953, 0.0042835088),
    # Wind direction in degrees clockwise from north.
    # Thus min set to 0 and max set to 360.
    'th': (0.0, 360.0, 190.32976, 72.59854),
    # Min/max temperature in Kelvin.
    # -20 degree C, 99.9 percentile
    'tmmn': (253.15, 298.94891357421875, 281.08768, 8.982386),
    # -20 degree C, 99.9 percentile
    'tmmx': (253.15, 315.09228515625, 295.17383, 9.815496),
    # Wind speed in m/s.
    # Negative values do not make sense, given there is a wind direction.
    # 0., 99.9 percentile
    'vs': (0.0, 10.024310074806237, 3.8500874, 1.4109988),
    # NFDRS fire danger index energy release component expressed in BTU's per
    # square foot.
    # Negative values do not make sense. Thus min set to zero.
    # 0., 99.9 percentile
    'erc': (0.0, 106.24891662597656, 37.326267, 20.846027),
    # Population density
    # min, 99.9 percentile
    'population': (0.0, 2534.06298828125, 25.531384, 154.72331),
    # We don't want to normalize the FireMasks.
    # 1 indicates fire, 0 no fire, -1 unlabeled data
    'PrevFireMask': (-1.0, 1.0, 0.0, 1.0),
    'FireMask': (-1.0, 1.0, 0.0, 1.0),
}

"""Dataset reader for Earth Engine data."""


def _get_base_key(key: Text) -> Text:
    """Extracts the base key from the provided key.

    Earth Engine exports TFRecords containing each data variable with its
    corresponding variable name. In the case of time sequences, the name of the
    data variable is of the form 'variable_1', 'variable_2', ..., 'variable_n',
    where 'variable' is the name of the variable, and n the number of elements
    in the time sequence. Extracting the base key ensures that each step of the
    time sequence goes through the same normalization steps.
    The base key obeys the following naming pattern: '([a-zA-Z]+)'
    For instance, for an input key 'variable_1', this function returns 'variable'.
    For an input key 'variable', this function simply returns 'variable'.

    Args:
      key: Input key.

    Returns:
      The corresponding base key.

    Raises:
      ValueError when `key` does not match the expected pattern.
    """
    match = re.match(r'([a-zA-Z]+)', key)
    if match:
        return match.group(1)
    raise ValueError(
        'The provided key does not match the expected pattern: {}'.format(key)
    )


def _clip_and_rescale(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and rescales inputs with the stats corresponding to `key`.

    Args:
      inputs: Inputs to clip and rescale.
      key: Key describing the inputs.

    Returns:
      Clipped and rescaled input.

    Raises:
      ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(
            'No data statistics available for the requested key: {}.'.format(key)
        )
    min_val, max_val, _, _ = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    return tf.math.divide_no_nan((inputs - min_val), (max_val - min_val))


def _clip_and_normalize(inputs: tf.Tensor, key: Text) -> tf.Tensor:
    """Clips and normalizes inputs with the stats corresponding to `key`.

    Args:
      inputs: Inputs to clip and normalize.
      key: Key describing the inputs.

    Returns:
      Clipped and normalized input.

    Raises:
      ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in DATA_STATS:
        raise ValueError(
            'No data statistics available for the requested key: {}.'.format(key)
        )
    min_val, max_val, mean, std = DATA_STATS[base_key]
    inputs = tf.clip_by_value(inputs, min_val, max_val)
    inputs = inputs - mean
    return tf.math.divide_no_nan(inputs, std)


def _get_features_dict(
    sample_size: int,
    features: List[Text],
) -> Dict[Text, tf.io.FixedLenFeature]:
    """Creates a features dictionary for TensorFlow IO.

    Args:
      sample_size: Size of the input tiles (square).
      features: List of feature names.

    Returns:
      A features dictionary for TensorFlow IO.
    """
    sample_shape = [sample_size, sample_size]
    features = set(features)
    columns = [
        tf.io.FixedLenFeature(shape=sample_shape, dtype=tf.float32) for _ in features
    ]
    return dict(zip(features, columns))


def _parse_fn(
    example_proto: tf.train.Example,
    data_size: int,
    sample_size: int,
    num_in_channels: int,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
    center_crop: bool,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads a serialized example.

    Args:
      example_proto: A TensorFlow example protobuf.
      data_size: Size of tiles (square) as read from input files.
      sample_size: Size the tiles (square) when input into the model.
      num_in_channels: Number of input channels.
      clip_and_normalize: True if the data should be clipped and normalized.
      clip_and_rescale: True if the data should be clipped and rescaled.
      random_crop: True if the data should be randomly cropped.
      center_crop: True if the data should be cropped in the center.

    Returns:
      (input_img, output_img) tuple of inputs and outputs to the ML model.
    """
    if random_crop and center_crop:
        raise ValueError('Cannot have both random_crop and center_crop be True')
    input_features, output_features = INPUT_FEATURES, OUTPUT_FEATURES
    feature_names = input_features + output_features
    features_dict = _get_features_dict(data_size, feature_names)
    features = tf.io.parse_single_example(example_proto, features_dict)

    if clip_and_normalize:
        inputs_list = [
            _clip_and_normalize(features.get(key), key) for key in input_features
        ]
    elif clip_and_rescale:
        inputs_list = [
            _clip_and_rescale(features.get(key), key) for key in input_features
        ]
    else:
        inputs_list = [features.get(key) for key in input_features]

    inputs_stacked = tf.stack(inputs_list, axis=0)
    input_img = tf.transpose(inputs_stacked, [1, 2, 0])

    outputs_list = [features.get(key) for key in output_features]
    assert outputs_list, 'outputs_list should not be empty'
    outputs_stacked = tf.stack(outputs_list, axis=0)

    outputs_stacked_shape = outputs_stacked.get_shape().as_list()
    assert len(outputs_stacked.shape) == 3, (
        'outputs_stacked should be rank 3'
        'but dimensions of outputs_stacked'
        f' are {outputs_stacked_shape}'
    )
    output_img = tf.transpose(outputs_stacked, [1, 2, 0])

    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(
            input_img, output_img, sample_size, num_in_channels, 1
        )
    if center_crop:
        input_img, output_img = center_crop_input_and_output_images(
            input_img, output_img, sample_size
        )
    return input_img, output_img


def get_dataset(
    file_pattern: Text,
    data_size: int,
    sample_size: int,
    batch_size: int,
    num_in_channels: int,
    compression_type: Text,
    clip_and_normalize: bool,
    clip_and_rescale: bool,
    random_crop: bool,
    center_crop: bool,
) -> tf.data.Dataset:
    """Gets the dataset from the file pattern.

    Args:
      file_pattern: Input file pattern.
      data_size: Size of tiles (square) as read from input files.
      sample_size: Size the tiles (square) when input into the model.
      batch_size: Batch size.
      num_in_channels: Number of input channels.
      compression_type: Type of compression used for the input files.
      clip_and_normalize: True if the data should be clipped and normalized, False
        otherwise.
      clip_and_rescale: True if the data should be clipped and rescaled, False
        otherwise.
      random_crop: True if the data should be randomly cropped.
      center_crop: True if the data shoulde be cropped in the center.

    Returns:
      A TensorFlow dataset loaded from the input file pattern, with features
      described in the constants, and with the shapes determined from the input
      parameters to this function.
    """
    if clip_and_normalize and clip_and_rescale:
        raise ValueError('Cannot have both normalize and rescale.')
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=compression_type),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x: _parse_fn(  # pylint: disable=g-long-lambda
            x,
            data_size,
            sample_size,
            num_in_channels,
            clip_and_normalize,
            clip_and_rescale,
            random_crop,
            center_crop,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    print(dataset.element_spec)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def random_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly axis-align crop input and output image tensors.

    Args:
      input_img: Tensor with dimensions HWC.
      output_img: Tensor with dimensions HWC.
      sample_size: Side length (square) to crop to.
      num_in_channels: Number of channels in `input_img`.
      num_out_channels: Number of channels in `output_img`.
    Returns:
      input_img: Tensor with dimensions HWC.
      output_img: Tensor with dimensions HWC.
    """
    combined = tf.concat([input_img, output_img], axis=2)
    combined = tf.image.random_crop(
        combined, [sample_size, sample_size, num_in_channels + num_out_channels]
    )
    input_img = combined[:, :, 0:num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img


def center_crop_input_and_output_images(
    input_img: tf.Tensor,
    output_img: tf.Tensor,
    sample_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calls `tf.image.central_crop` on input and output image tensors.

    Args:
      input_img: Tensor with dimensions HWC.
      output_img: Tensor with dimensions HWC.
      sample_size: Side length (square) to crop to.
    Returns:
      input_img: Tensor with dimensions HWC.
      output_img: Tensor with dimensions HWC.
    """
    central_fraction = sample_size / input_img.shape[0]
    input_img = tf.image.central_crop(input_img, central_fraction)
    output_img = tf.image.central_crop(output_img, central_fraction)
    return input_img, output_img


def calculate_fire_change(
    prev_fire_mask: tf.Tensor,
    fire_mask: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calculates the change in fire pixels between two fire masks
    and outputs a mask of pixels with three values: 0, 1, 2

    Args:
      prev_fire_mask: Tensor, where 0 indicates no fire and 1 indicates fire
      fire_mask: Tensor, where 0 indicates no fire and 1 indicates fire
    Returns:
      fire_change_mask: Tensor
    """
    fire_change_mask = tf.subtract(fire_mask, prev_fire_mask)
    fire_change_mask = tf.where(fire_change_mask < -1, -1, fire_change_mask)
    fire_change_mask = tf.where(fire_change_mask > 1, 1, fire_change_mask)
    # add 1 to the fire change mask to get values of 0, 1, 2
    fire_change_mask = tf.add(fire_change_mask, 1)
    return fire_change_mask


def random_crop(img: np.array, target: np.array, crop_size: int = 32):
    """
    Randomly crops a numpy array to the specified size.

    Args:
        img (numpy array): The input image to crop.
        crop_size (tuple): The desired output size of the crop.

    Returns:
        numpy array: The randomly cropped image.
    """
    # Input img batch shape is (C, H, W)
    h, w = img.shape[1:]
    # Calculate the random offsets for the crop
    top = np.random.randint(0, h - crop_size)
    left = np.random.randint(0, w - crop_size)

    # Crop the image using the calculated offsets
    img = img[:, top : top + crop_size, left : left + crop_size]
    target = target[:, top : top + crop_size, left : left + crop_size]

    return img, target
