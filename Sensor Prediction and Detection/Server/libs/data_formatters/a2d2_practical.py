# coding=utf-8
# Lint as: python3
"""Custom formatting functions for A2D2 H-index dataset.

Defines dataset specific column definitions and data transformations.
"""
import pandas as pd
import libs.data_formatters.base 
import libs.libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = libs.data_formatters.base.GenericDataFormatter
DataTypes = libs.data_formatters.base.DataTypes
InputTypes = libs.data_formatters.base.InputTypes

class A2D2PracticalFormatter(GenericDataFormatter):
  """Defines and formats data for the A2D2 H-index dataset.

  Note that per-entity z-score normalization is used here, and is implemented
  across functions.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """
  _column_definition = [
      ('time_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('health_index', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

  def __init__(self):
    """Initializes formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']

  def split_data(self, df, valid_boundary=2, test_boundary=3):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    index = df['categorical_id']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
    test = df.loc[index >= test_boundary]

    self.set_scalers(df)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):

      if len(sliced) >= self._time_steps:
        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(data)

        self._target_scaler[identifier] \
      = sklearn.preprocessing.StandardScaler().fit(targets)
      identifiers.append(identifier)

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,
                                                column_definitions)
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Transform real inputs per entity
    df_list = []
    for identifier, sliced in df.groupby(id_col):
      #print('-------------')
      #print(identifier)
      #print(sliced)
      
      # Filter out any trajectories that are too short 
      if len(sliced) >= self._time_steps:
        #print('if passed')
        sliced_copy = sliced.copy()
        #print(self._real_scalers)
        sliced_copy[real_inputs] = self._real_scalers[identifier].transform(
            sliced_copy[real_inputs].values)
        df_list.append(sliced_copy)
        #print('slice appended to df_list')
      #print(df_list)
    #print('Time to concat')
    output = pd.concat(df_list, axis=0)
    #print('did it')
    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of un-normalized predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      target_scaler = self._target_scaler[identifier]

      for col in column_names:
        if col not in {'forecast_time', 'identifier'}:
          sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col])
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 150, # 15*60/2.4 (15min)100
        'num_encoder_steps': 100, # (3min)60
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimized model parameters."""

    model_params = {
        'dropout_rate': 0.4,
        'hidden_layer_size': 40,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 1,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return -1, -1
