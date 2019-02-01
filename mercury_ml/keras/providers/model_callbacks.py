from keras.callbacks import Callback, TensorBoard, BaseLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN, \
        ProgbarLogger, RemoteMonitor, LearningRateScheduler, ReduceLROnPlateau, CSVLogger

import os

# from ml_workflow.utils import utils
# from ml_workflow.containers.core import Core
# from sklearn.metrics import roc_auc_score
#
# class AUCGeneratorCallBackProvider(Callback):
#     """ Class for AUC callback in Keras """
#     def __init__(self, params):
#         super(AUCGeneratorCallBackProvider, self).__init__()
#         self.generator = Core.session.datasets()[params["dataset_to_use"]]["data_formats"]["ml"][params["generator_to_use"]]
#         self.steps = params["steps"]
#         self.verbose = params["verbose"]
#         self.dataset = params["dataset_to_use"]
#
#     def on_epoch_end(self, epoch, logs={}):
#         """
#         A function that defines a Callback that should be run at the end of a training epoch
#         :param epoch:
#         :param logs:
#         :return:
#         """
#         # Get probability matrix
#         predictions = self.model.predict_generator(
#             generator=self.generator,
#             steps=self.steps,
#             verbose=self.verbose)
#
#         # Target matrix
#
#         targets = self.generator.get_target_dummies()[:self.generator.batch_size*self.steps,:]
#         # macro average AUC
#         try:
#             auc_macro = roc_auc_score(y_true=targets, y_score=predictions, average="macro")
#             print(" - {}_auc: ".format(self.dataset) + str(round(auc_macro, 4)))
#         except:
#             print(" - {}_auc: calculation failed")
#

class ModelCheckpointProvider(ModelCheckpoint):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(ModelCheckpointProvider, self).__init__(**params)
        model_dir = os.path.dirname(params["filepath"])
        model_dir = self._make_local_path(model_dir)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

    def _make_local_path(self, path_name):
        if path_name[0] == ".":
            path_name = os.path.join(os.getcwd(), path_name)
            path_name = os.path.abspath(path_name)
        return path_name

class TensorBoardProvider(TensorBoard):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(TensorBoardProvider, self).__init__(**params)

class BaseLoggerProvider(BaseLogger):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(BaseLoggerProvider, self).__init__(**params)


class EarlyStoppingProvider(EarlyStopping):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(EarlyStoppingProvider, self).__init__(**params)


class TerminateOnNaNProvider(TerminateOnNaN):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(TerminateOnNaNProvider, self).__init__()


class ProgbarLoggerProvider(ProgbarLogger):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(ProgbarLoggerProvider, self).__init__(**params)


class CSVLoggerProvider(CSVLogger):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(CSVLoggerProvider, self).__init__(**params)


class ReduceLROnPlateauProvider(ReduceLROnPlateau):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(ReduceLROnPlateauProvider, self).__init__(**params)


class LearningRateSchedulerProvider(LearningRateScheduler):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(LearningRateSchedulerProvider, self).__init__(**params)


class RemoteMonitorProvider(RemoteMonitor):
    """ Class for AUC callback in Keras """
    def __init__(self, params):
        super(RemoteMonitorProvider, self).__init__(**params)
