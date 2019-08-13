"""
This contains (failed) attempts at creating Keras models for the BPNN
"""
# Atomic model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import glorot_uniform

#Class Seg

def create_atomic_model(bfsize, nodes=[48,32]):
    """
    This is a Keras model for one of the elements for a BPNN. These elements
    would need to be hooked up later.
    """
    basis = Input(shape=(bfsize,))
    seg_ids = Input(shape=(None,))
    num_layers = len(nodes)
    model = tf.keras.Sequential()
    model.add(Dense(nodes[0], activation='tanh',
		    kernel_initializer=glorot_uniform,
                    bias_initializer=glorot_uniform)(basis))
    return model
