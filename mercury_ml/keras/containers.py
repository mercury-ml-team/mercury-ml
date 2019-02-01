"""
Simple IoC containers that provide direct access to various Keras providers
"""


class ModelSavers:
    from mercury_ml.keras.providers import model_saving
    save_hdf5 = model_saving.save_keras_hdf5
    save_tensorflow_graph = model_saving.save_tensorflow_graph
    save_tensorrt_pbtxt_config = model_saving.save_tensorrt_pbtxt_config
    save_tensorrt_json_config = model_saving.save_tensorrt_json_config
    save_labels_txt = model_saving.save_labels_txt
    save_tensorflow_serving_predict_signature_def = model_saving.save_tensorflow_serving_predict_signature_def

class ModelLoaders:
    from mercury_ml.keras.providers import model_loading
    load_hdf5 = model_loading.load_hdf5_model

class LossFunctionFetchers:
    from mercury_ml.keras.providers import loss_function_fetching
    get_keras_loss = loss_function_fetching.get_keras_loss
    get_custom_loss = loss_function_fetching.get_custom_loss

class OptimizerFetchers:
    from mercury_ml.keras.providers import optimizer_fetching
    get_keras_optimizer = optimizer_fetching.get_keras_optimizer

class ModelCompilers:
    from mercury_ml.keras.providers import model_compilation
    compile_model = model_compilation.compile_model


class ModelFitters:
    from mercury_ml.keras.providers import model_fitting
    fit = model_fitting.fit
    fit_generator = model_fitting.fit_generator

class ModelDefinitions:
    from mercury_ml.keras.providers.model_definition import conv_simple, mlp_simple

    # these are just two small example model definitions. Users should define their own models
    # to use as follows:
    # >>> ModelDefinitions.my_model = my_model_module.define_model

    define_conv_simple = conv_simple.define_model
    define_mlp_simple = mlp_simple.define_model


class GeneratorPreprocessingFunctionGetters:
    from mercury_ml.keras.providers.generator_preprocessors import get_random_eraser
    get_random_eraser = get_random_eraser


class CallBacks:
    from mercury_ml.keras.providers.model_callbacks import  TensorBoardProvider, \
        BaseLoggerProvider, EarlyStoppingProvider, ModelCheckpointProvider, TerminateOnNaNProvider, \
        ProgbarLoggerProvider, RemoteMonitorProvider, LearningRateSchedulerProvider, ReduceLROnPlateauProvider, \
        CSVLoggerProvider

    tensorboard = TensorBoardProvider
    base_logger = BaseLoggerProvider
    terminate_on_nan = TerminateOnNaNProvider
    progbar_logger = ProgbarLoggerProvider
    model_checkpoint = ModelCheckpointProvider
    early_stopping = EarlyStoppingProvider
    remote_monitor = RemoteMonitorProvider
    learning_rate_scheduler = LearningRateSchedulerProvider
    reduce_lr_on_plateau = ReduceLROnPlateauProvider
    csv_logger = CSVLoggerProvider

class ModelEvaluators:
    from mercury_ml.keras.providers import model_evaluation
    evaluate = model_evaluation.evaluate
    evaluate_generator = model_evaluation.evaluate_generator

class PredictionFunctions:
    from mercury_ml.keras.providers import prediction
    predict = prediction.predict
    predict_generator = prediction.predict_generator
