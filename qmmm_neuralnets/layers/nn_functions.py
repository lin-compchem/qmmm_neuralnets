import math
import tensorflow as tf


def weight_variable(shape, dtype=tf.float32):
    """Create a weight variable with appropriate initialization."""
    initial = tf.random.truncated_normal(shape, stddev=0.1, dtype=dtype)
    return tf.Variable(initial, dtype=dtype)


def xavier_variable(shape, dtype=tf.float32):
    """
    Create a weight variable initialized with Xavier and Yoshua's method
    """
    initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    return tf.Variable(initializer(shape))


def var_scaled_variable(shape, dtype=tf.float32):
    """
    vscaled_variable
    (variance_scaled_variabler)
    Create an initializer that does the following {From TF website}
    
  if mode='FAN_IN': # Count only number of input connections.
    n = fan_in
  elif mode='FAN_OUT': # Count only number of output connections.
    n = fan_out
  elif mode='FAN_AVG': # Average number of inputs and output connections.
    n = (fan_in + fan_out)/2.0

    truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(dtype=dtype)
    return tf.Variable(initializer(shape))


def sigmoid_scaled_variable(shape, dtype=tf.float32, seed=None):
    """
    scale the variables by the distribution in 
    """
    limit = math.sqrt(6./shape[0]) * 4
    return tf.random_ops.random_uniform(shape, -limit, limit, dtype, seed=seed)


def he_normal_variable(shape, dtype=tf.float32):
    """
    He normal variable
    
    returns variables with normal distribution centered on 0 and with
    stddev = sqrt(2 / fan_in)

    """
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=dtype)
    return tf.Variable(initializer(shape))

def selu_normal_variable(shape, dtype=tf.float32):
    """
    He normal variable
    
    returns variables with normal distribution centered on 0 and with
    stddev = sqrt(2 / fan_in)

    """
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=dtype)
    return tf.Variable(initializer(shape), dtype=dtype)


def bias_variable(shape, dtype=tf.float32):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = var_scaled_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = var_scaled_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        
        return activations

def xavier_sigmoid_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = sigmoid_scaled_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = sigmoid_scaled_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = tf.nn.sigmoid(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        
        return activations

def xavier_dense_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.tanh):
    """Reusable code for making a simple neural net layer.
    Use scaling for sigmoid from Xavier and Youseph
    
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = xavier_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = xavier_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        
        return activations
    
def he_normal_dense_layer(input_tensor, input_dim, output_dim, layer_name,
                          act=tf.nn.relu, dtype=tf.float32):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = selu_normal_variable([input_dim, output_dim], dtype)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = selu_normal_variable([output_dim], dtype)
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        
        return activations
 
def selu_dense_layer(input_tensor, input_dim, output_dim, layer_name):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = he_normal_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = he_normal_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        
        return activations
 
def nn_out_layer(input_tensor, input_dim, output_dim, layer_name,
                 dtype=tf.float32):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    
    It doesn't have an activation function
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim], dtype=dtype)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim], dtype=dtype)
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        return preactivate
