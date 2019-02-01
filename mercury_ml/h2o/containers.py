"""
Simple IoC containers that provide direct access to various H2O providers
"""


class ModelDefinitions:

    from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator
    from h2o.estimators.deepwater import H2ODeepWaterEstimator
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    from h2o.estimators.glm import H2OGeneralizedLinearEstimator
    from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
    from h2o.estimators.kmeans import H2OKMeansEstimator
    from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
    from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
    from h2o.estimators.random_forest import H2ORandomForestEstimator
    from h2o.estimators.svd import H2OSingularValueDecompositionEstimator
    
    rf = H2ORandomForestEstimator
    mlp = H2ODeepLearningEstimator
    deep_water = H2ODeepWaterEstimator
    auto_encoder = H2OAutoEncoderEstimator
    gbm = H2OGradientBoostingEstimator
    glm = H2OGeneralizedLinearEstimator
    low_rank = H2OGeneralizedLowRankEstimator
    k_means = H2OKMeansEstimator
    naive_bayes = H2ONaiveBayesEstimator
    pca = H2OPrincipalComponentAnalysisEstimator
    svd = H2OSingularValueDecompositionEstimator


class ModelFitters:
    from mercury_ml.h2o.providers import model_fitting

    fit = model_fitting.fit

class ModelSavers:
    from mercury_ml.h2o.providers import model_saving

    save_h2o_model = model_saving.save_h2o_model
    save_json_details = model_saving.save_model_details
    save_pojo = model_saving.save_pojo
    save_mojo = model_saving.save_mojo
    save_pojo_jar = model_saving.save_pojo_jar
    save_mojo_jar = model_saving.save_mojo_jar

class ModelLoaders:
    from mercury_ml.h2o.providers import model_loading
    load_h2o_model = model_loading.load_h2o_model
    load_mojo = model_loading.load_mojo_model


class ModelEvaluators:
    from mercury_ml.h2o.providers import model_evaluation

    evaluate = model_evaluation.evaluate
    evaluate_threshold_metrics = model_evaluation.evaluate_threshold_metrics


class PredictionFunctions:
    from mercury_ml.h2o.providers import prediction

    predict = prediction.predict


class SessionInitiators:
    from mercury_ml.h2o.providers import session

    get_or_create_h2o = session.get_or_create_h2o
    get_or_create_h2o_sparkling = session.get_or_create_h2o_sparkling