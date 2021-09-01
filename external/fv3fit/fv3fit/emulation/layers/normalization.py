import abc
import tensorflow as tf


class NormLayer(tf.keras.layers.Layer, abc.ABC):
    def __init__(self, name=None, **kwargs):
        super(NormLayer, self).__init__(name=name)
        self.fitted = False

    @abc.abstractmethod
    def _build_mean(self, in_shape):
        self.mean = None

    @abc.abstractmethod
    def _build_sigma(self, in_shape):
        self.sigma = None

    def build(self, in_shape):
        self._build_mean(in_shape)
        self._build_sigma(in_shape)

    @abc.abstractmethod
    def _fit_mean(self, tensor):
        pass

    @abc.abstractmethod
    def _fit_sigma(self, tensor):
        pass

    def fit(self, tensor):
        self(tensor)
        self._fit_mean(tensor)
        self._fit_sigma(tensor)
        self.fitted = True

    @abc.abstractmethod
    def call(self, tensor) -> tf.Tensor:
        pass


class PerFeatureMean(NormLayer):
    """
    Build layer weights and fit a mean value for each
    feature in a tensor (assumed first dimension is samples).
    """

    def _build_mean(self, in_shape):
        self.mean = self.add_weight(
            "mean", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )

    def _fit_mean(self, tensor):
        self.mean.assign(tf.cast(tf.reduce_mean(tensor, axis=0), tf.float32))


class PerFeatureStd(NormLayer):
    """
    Build layer weights and fit a standard deviation value [sigma]
    for each feature in a tensor (assumed first dimension is samples).
    """

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[in_shape[-1]], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        self.sigma.assign(tf.cast(tf.math.reduce_std(tensor, axis=0), tf.float32))


class FeatureMaxStd(NormLayer):
    """
    Build layer weights and fit a standard deviation value based
    on the maximum of all features in a tensor (assumed first
    dimension is samples).
    """

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        stddev = tf.math.reduce_std(tensor, axis=0)
        max_std = tf.cast(tf.reduce_max(stddev), tf.float32)
        self.sigma.assign(max_std)


class FeatureAvgStd(NormLayer):
    """
    Build layer weights and fit a standard deviation value based
    on the average of all features in a tensor (assumed first
    dimension is samples).
    """

    def _build_sigma(self, in_shape):
        self.sigma = self.add_weight(
            "sigma", shape=[], dtype=tf.float32, trainable=False
        )

    def _fit_sigma(self, tensor):
        stddev = tf.math.reduce_std(tensor, axis=0)
        avg_std = tf.cast(tf.reduce_mean(stddev), tf.float32)
        self.sigma.assign(avg_std)


class StandardNormLayer(PerFeatureMean, PerFeatureStd):
    """
    Normalization layer that removes mean and standard
    deviation for each feature individually.

    Args:
        epsilon: Floating point  floor added to sigma prior to
            division
    """

    def __init__(self, epsilon: float = 1e-7, name=None):
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, tensor):
        return (tensor - self.mean) / (self.sigma + self.epsilon)


class StandardDenormLayer(PerFeatureMean, PerFeatureStd):
    """
    De-normalization layer that scales by the standard
    deviation and adds the mean for each feature individually.
    """

    def call(self, tensor):
        return tensor * self.sigma + self.mean


class MaxFeatureStdNormLayer(PerFeatureMean, FeatureMaxStd):
    """
    Normalization layer that removes mean for each feature
    individually but scales all features by the maximum standard
    deviation calculated over all features. Useful to preserve
    feature scale relationships.
    """

    def call(self, tensor):
        return (tensor - self.mean) / self.sigma


class MaxFeatureStdDenormLayer(MaxFeatureStdNormLayer):
    """
    De-normalization layer that scales all features by the maximum
    standard deviation calculated over all features and adds back
    the mean for each individual feature.
    """

    def call(self, tensor):
        return tensor * self.sigma + self.mean
    

class AvgFeatureStdNormLayer(PerFeatureMean, FeatureAvgStd):
    """
    Normalization layer that removes mean for each feature
    individually but scales all features by the average standard
    deviation calculated over all features. Useful to preserve
    feature scale relationships.
    """

    def call(self, tensor):
        return (tensor - self.mean) / self.sigma


class AvgFeatureStdDenormLayer(AvgFeatureStdNormLayer):
    """
    De-normalization layer that scales all features by the average
    standard deviation calculated over all features and adds back
    the mean for each individual feature.
    """

    def call(self, tensor):
        return tensor * self.sigma + self.mean
