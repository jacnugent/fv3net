from typing import Optional
import tensorflow as tf


class CombineInputs(tf.keras.layers.Layer):
    """Input tensor stacking with option to add a dimension for RNNs"""

    def __init__(
        self, combine_axis: int, *args, expand_axis: Optional[int] = None, **kwargs
    ):
        """
        Args:
            combine_axis: Axis to concatenate tensors along.  Note that if expand_axis
                is specified, it is applied before concatenation.  E.g., combine_axis=1
                and expand_axis=1 will concatenate along the newly created dimension.
            expand_axis: New axis to add to the input tensors
        """
        super().__init__(*args, **kwargs)

        self.combine_axis = combine_axis
        self.expand_axis = expand_axis

    def call(self, inputs):

        if self.expand_axis is not None:
            inputs = [
                tf.expand_dims(tensor, axis=self.expand_axis) for tensor in inputs
            ]

        return tf.concat(inputs, axis=self.combine_axis)


class RNNBlock(tf.keras.layers.Layer):
    """
    RNN connected to an MLP for prediction

    Layer call expects a 3-D input tensor (sample, z, variable),
    which recurses backwards (top-to-bottom for the physics-param data)
    by default over the z dimension.
    """

    def __init__(
        self,
        *args,
        channels: int = 256,
        dense_width: int = 256,
        dense_depth: int = 1,
        activation: str = "relu",
        go_backwards: bool = True,
        **kwargs,
    ):
        """
        Args:
            channels: width of RNN layer
            dense_width: width of MLP layer, only used if dense_depth > 0
            dense_depth: number of MLPlayers connected to RNN output, set to
                0 to disable
            activation: activation function to use for RNN and MLP
            go_backwards: whether to recurse in the reverse direction
                over the dim 1.  If using wrapper outputs, TOA starts
                at 0, should be false
        """
        super().__init__(*args, **kwargs)

        self.rnn = tf.keras.layers.SimpleRNN(
            channels, activation=activation, go_backwards=go_backwards
        )
        self.dense = MLPBlock(width=dense_width, depth=dense_depth)

    def call(self, input):
        """
        Args:
            input: a 3-d tensor with dims (sample, per_var_features, variables)
        """

        rnn_out = self.rnn(input)

        output = self.dense(rnn_out)

        return output


class MLPBlock(tf.keras.layers.Layer):
    """MLP layer for basic NN predictions"""

    def __init__(
        self,
        *args,
        width: int = 256,
        depth: int = 2,
        activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            width: width of MLP layer, only used if dense_depth > 0
            depth: number of MLPlayers, set to 0 to disable
            activation: activation function to use for RNN and MLP
        """
        super().__init__(*args, **kwargs)

        self.dense = [
            tf.keras.layers.Dense(width, activation=activation) for i in range(depth)
        ]

    def call(self, input):

        outputs = input

        for i in range(len(self.dense)):
            outputs = self.dense[i](outputs)

        return outputs


def NoWeightSharingSLP(
    n: int = 1, width: int = 128, activation="relu"
) -> tf.keras.Model:
    """Multi-output model which doesn't share any weights between outputs

    Args:
        n: number of outputs
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(n * width, activation=activation),
            tf.keras.layers.Reshape([n, width]),
            tf.keras.layers.LocallyConnected1D(1, 1),
            tf.keras.layers.Flatten(),
        ]
    )