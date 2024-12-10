import os
import numpy as np
from typing import Any, Type
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from sklearn.pipeline import Pipeline
import joblib

class SklearnPipelineMaterializer(BaseMaterializer):
    """
    ZenML materializer for scikit-learn Pipeline objects containing XGBoost regressors.
    This materializer handles the serialization and deserialization of Pipeline objects
    that include XGBoost regression models as steps. It uses joblib for efficient
    storage and loading of the pipeline objects.

    Attributes: 
    ASSOCIATED_TYPES : tuple - Types associated with this materializer. In this case, scikit-learn Pipeline.
    ASSOCIATED_ARTIFACT_TYPE : ArtifactType - The type of artifact being handled, which is a model.
    """
    ASSOCIATED_TYPES = (Pipeline,)  
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL 

    def load(self, data_type: Type[Pipeline]) -> Pipeline:
        """
        Load a scikit-learn Pipeline from the artifact store.

        Parameters: 
        data_type : Type[Pipeline] - The type of data being loaded. This must match the associated type.

        Returns: 
        Pipeline : The loaded scikit-learn Pipeline object.
        """
        pipeline_path = os.path.join(self.uri, 'model.joblib')
        return joblib.load(pipeline_path)

    def save(self, pipeline: Pipeline) -> None:
        """
        Save a scikit-learn Pipeline to the artifact store.

        Parameters: 
        pipeline : Pipeline - The scikit-learn Pipeline object to save.

        Returns: 
        None
        """
        pipeline_path = os.path.join(self.uri, 'model.joblib')
        joblib.dump(pipeline, pipeline_path)

class NumpyInt64Materializer(BaseMaterializer):
    """
    Materializer to handle the saving and loading of numpy int64 objects in the ZenML framework.

    Attributes: 
    ASSOCIATED_TYPES : tuple - Types associated with this materializer. In this case, numpy int64.
    ASSOCIATED_ARTIFACT_TYPE : ArtifactType - The type of artifact being handled, which is data.
    """
    ASSOCIATED_TYPES = (np.int64,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[np.int64]) -> np.int64:
        """
        Load a numpy int64 object from the artifact store.

        Parameters: 
        data_type : Type[np.int64] - The type of data being loaded. This must match the associated type.

        Returns: 
        np.int64 - The loaded numpy int64 object.
        """ 
        int_path = os.path.join(self.uri, 'numpy_int64.npy')
        return np.load(int_path)

    def save(self, data: np.int64) -> None:
        """
        Save a numpy int64 object to the artifact store.

        Parameters: 
        data : np.int64 - The numpy int64 object to save.

        Returns: 
        None
        """
        int_path = os.path.join(self.uri, 'numpy_int64.npy')
        np.save(int_path, data) 