import tensorflow.compat.v1 as tf
import libs.data_formatters.base
import libs.expt_settings.configs
import libs.libs.hyperparam_opt
import libs.libs.tft_model
import libs.libs.utils as utils
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ExperimentConfig = libs.expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.libs.tft_model.TemporalFusionTransformer


class predictNet():
    def __init__(self):
        self.first_predict = True
        self.folder = "./libs/predModel"
        self.config = ExperimentConfig(
            experiment='a2d2', root_folder=self.folder)
        self.modelFolder = self.config.model_folder
        self.dataFormatter = self.config.make_data_formatter()
        self.default_keras_session = tf.keras.backend.get_session()
        self.gpuAvailable = True
        if self.gpuAvailable:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device='gpu', gpu_id=0)
        else:
            self.tf_config = utils.get_default_tensorflow_config(
                tf_device='cpu')
        print("loading Pretrained Models")
        self.fixedParams = self.dataFormatter.get_experiment_params()
        self.params = self.dataFormatter.get_default_model_params()
        self.params["model_folder"] = self.modelFolder
        self.optManager = HyperparamOptManager(
            {k: [self.params[k]] for k in self.params}, self.fixedParams, self.modelFolder)
        self.succ = self.optManager.load_results()
        self.bestParams = self.optManager.get_best_params()
        tf.reset_default_graph()
        tf.Graph().as_default()
        self.bestParams['category_counts'] = [3]
        self.sess = tf.Session(config=self.tf_config)
        tf.keras.backend.set_session(self.sess)
        self.model = ModelClass(self.bestParams, use_cudnn='yes')
        self.model.load("./libs/predModel/saved_models/a2d2")

    def __del__(self):
        self.sess.close()

    def dataProcessing(self, HI):
        HI['health_index'][len(HI['health_index']) - 150:len(HI['health_index']) - 51] = savgol_filter(
            HI['health_index'][len(HI['health_index']) - 150:len(HI['health_index']) - 51], 53, 3)
        #print(HI[100:])
        if self.first_predict:
            self.dataFormatter.set_scalers(HI)
            self.first_predict = False
        ready = self.dataFormatter.transform_inputs(HI)
        return ready

    def predict(self, HI):
        dataFrame = self.dataProcessing(HI)
        prediction = self.model.predictLive(dataFrame)
        p10Prediction = self.dataFormatter.format_predictions(
            prediction['p10'])
        p50Prediction = self.dataFormatter.format_predictions(
            prediction['p50'])
        p90Prediction = self.dataFormatter.format_predictions(
            prediction['p90'])
        return p10Prediction, p50Prediction, p90Prediction
