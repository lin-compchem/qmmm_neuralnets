"""
This file contains the tensorflow implementation of the BPSF functions for a
BPSF NN
"""
import tensorflow as tf
import pickle

from qmmm_neuralnets.layers.nn_functions import he_normal_dense_layer, nn_out_layer


class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


share_variables = lambda func: tf.make_template(
    func.__name__, func, create_scope_now_=True)


def define_config(xh, xo):
    """
    This subroutine sets up a class-like object of input settings to the
    neural network. 

    Params
    ------
    xh: ndarray
        The hydrogen basis set
    xo: ndarray
        The oxygen basis set
    
    Returns
    -------
    config: object
        This object has all of the configuration settings.
    """
    config = AttrDict()
    config.hlen = xh.shape[1]
    config.olen = xo.shape[1]
    config.h_nodes = [128, 64]
    config.o_nodes = [128, 64]
    config.output_size = 1
    config.learning_rate = 1e-3
    config.save_name = './models/first_feat0_{0:06d}'
    config.bf_dtype = tf.float32
    config.y_dtype = tf.float32
    config.id_dtype = tf.int32
    config.optimizer = 'adam'
    config.energy_loss_scale = 1.0
    config.grad_loss_scale = 1/10./3.
    config.grad_atoms = 13
    config.export_path = 'saved_model'
    config.restore_path = None
    config.score_path = None
    config.builder = None
    return config


def element_nn(nodes, input_basis, input_segids, basis_dim, layer_name,
        act=tf.nn.tanh, dtype=tf.float32):
    """

    Parameters
    ----------
    nodes
    input_basis
    input_segids
    basis_dim
    layer_name
    act

    Returns
    -------

    """
    layer_type = he_normal_dense_layer
    out_type = nn_out_layer
    num_layers = len(nodes)
    layers = []
    layer_str = layer_name + '-%d'
    with tf.name_scope(layer_name):
        layers.append(layer_type(input_basis, basis_dim, nodes[0],
            layer_str % 0, act=act, dtype=dtype))
        for l in range(1, num_layers):
            print(l)
            layers.append(layer_type(layers[l-1], nodes[l-1], nodes[l],
                layer_str % l, act=act, dtype=dtype))
        l += 1
        layers.append(nn_out_layer(layers[l-1], nodes[l-1], 1, layer_str %l,
                                   dtype=dtype))
        return tf.math.segment_sum(layers[l], input_segids)
        

def define_graph(config):
    """
    The entire graph is defined here based on the options set in the config
    subroutine.

    Parameters
    ----------
    config: AttrDict
        This contains parameters to setup the network
    train_grads: bool
        Add graph elements for training the gradient? See Notes

    Returns
    -------
    graph: AttrDict
        A BPNN Tensorflow Graph

    Notes
    -----
    The loss function can be accessed by graph.cost

    A loss function for the gradient can be obtained by first passing in True
    for the parameter train_grads.

    """
    tf.reset_default_graph()

    # Input options
    xh = tf.placeholder(dtype=config.bf_dtype, shape=(None, config.hlen),
            name='h_bas')
    xo = tf.placeholder(dtype=config.bf_dtype, shape=(None, config.olen),
            name='o_bas')
    y = tf.placeholder(dtype=config.bf_dtype, shape=(None), name='en')
    h_ids = tf.placeholder(dtype=config.id_dtype, shape=(None), name='h_ids')
    o_ids = tf.placeholder(dtype=config.id_dtype, shape=(None), name='o_ids')
    grad_scale = tf.placeholder(dtype=config.bf_dtype, shape=(),
                                name='grad_scale')

    # For gradient training, we must pass in the
    h_bas_grads = tf.placeholder(dtype=config.bf_dtype,
                                 shape=(None, config.hlen, config.grad_atoms, 3),
                                 name='h_basis_grads')
    o_bas_grads = tf.placeholder(dtype=config.bf_dtype,
                                 shape=(None, config.olen, config.grad_atoms, 3),
                                 name='o_basis_grads')
    ref_grads = tf.placeholder(dtype=config.bf_dtype,
                               shape=(None, config.grad_atoms, 3),
                               name='h_reference_cartesian_gradients')


    # The BPNN
    h_en = element_nn(config.h_nodes, xh, h_ids, config.hlen, 'h_nn',
                      dtype=config.bf_dtype)
    o_en = element_nn(config.o_nodes, xo, o_ids, config.olen, 'o_nn',
                      dtype=config.bf_dtype)
    nn_en = tf.add(h_en, o_en)
    energy_cost = tf.reduce_mean(tf.math.squared_difference(nn_en,
                                                     tf.reshape(y, (-1,1))),
                          name='energy_cost')
    squared_error = tf.math.squared_difference(nn_en, tf.reshape(y, (-1,1)))
    difference = tf.subtract(nn_en, tf.reshape(y, (-1,1)))

    # The gradients of the neural network WRT the basis functions
    dnn_dh, dnn_do = tf.gradients(nn_en, [xh, xo])[0:2]

    # Tensor contraction to [basis_size, ngrum_atoms, 3]
    h_bas_cart_grads = tf.einsum('ijkl,ij->ikl', h_bas_grads, dnn_dh)
    # Here we go to [batch_size, num_atoms, 3]
    h_cart_grads = tf.math.segment_sum(h_bas_cart_grads, h_ids)

    o_bas_cart_grads = tf.einsum('ijkl,ij->ikl', o_bas_grads, dnn_do)
    o_cart_grads = tf.math.segment_sum(o_bas_cart_grads, o_ids)

    # This gives us the total correction gradient
    corr_grad = tf.add(h_cart_grads, o_cart_grads)
    grad_error = tf.math.squared_difference(corr_grad, ref_grads, name='grad_error')
    #
    # This is replaced by MSE above
    #ge
    # This gives us the error in gradient
#    grad_error = tf.subtract(corr_grad, ref_grads, name='grad_error')
    # We need the norm of the error in gradient along the axis of xyz
#    grad_norm = tf.norm(grad_error, ord='euclidean', axis=2, name='grad_norm')




    # Sum before reduce mean, because otherwise the 0 padded axes will
    # affect the meanc
    cart_sum = tf.reduce_sum(grad_error, axis=2, name='cart_grad_sum')
    geom_sum = tf.reduce_sum(grad_error, axis=1, name='geom_grad_sum')
    geom_mean = tf.reduce_mean(geom_sum, name='grad_geom_mean')
    grad_cost = tf.multiply(geom_mean, grad_scale, name='grad_cost')
    cost = tf.add(grad_cost, energy_cost, name='total_cost')

    # Optimizer
    optimizer = define_optimizer(config)

    # Training and statistics
    train_step = optimizer.minimize(energy_cost)
    train_grads = optimizer.minimize(grad_cost)
    train_all = optimizer.minimize(cost)

    # Saving info
    saver = tf.train.Saver(max_to_keep=100000)

    return AttrDict(locals())


def define_graph_quickgrad(config):
    """
    The entire graph is defined here based on the options set in the config
    subroutine.

    Parameters
    ----------
    config: AttrDict
        This contains parameters to setup the network
    train_grads: bool
        Add graph elements for training the gradient? See Notes

    Returns
    -------
    graph: AttrDict
        A BPNN Tensorflow Graph

    Notes
    -----
    The loss function can be accessed by graph.cost

    A loss function for the gradient can be obtained by first passing in True
    for the parameter train_grads.

    """
    tf.reset_default_graph()

    # Input options
    xh = tf.placeholder(dtype=config.bf_dtype, shape=(None, config.hlen),
            name='h_bas')
    xo = tf.placeholder(dtype=config.bf_dtype, shape=(None, config.olen),
            name='o_bas')
    y = tf.placeholder(dtype=config.bf_dtype, shape=(None), name='en')
    h_ids = tf.placeholder(dtype=config.id_dtype, shape=(None), name='h_ids')
    o_ids = tf.placeholder(dtype=config.id_dtype, shape=(None), name='o_ids')
    grad_scale = tf.placeholder(dtype=config.bf_dtype, shape=(),
                                name='grad_scale')

    # For gradient training, we must pass in the
    h_bas_grads = tf.placeholder(dtype=config.bf_dtype,
                                 shape=(None, config.hlen, config.grad_atoms, 3),
                                 name='h_basis_grads')
    o_bas_grads = tf.placeholder(dtype=config.bf_dtype,
                                 shape=(None, config.olen, config.grad_atoms, 3),
                                 name='o_basis_grads')
    ref_grads = tf.placeholder(dtype=config.bf_dtype,
                               shape=(None, config.grad_atoms, 3),
                               name='h_reference_cartesian_gradients')


    # The BPNN
    h_en = element_nn(config.h_nodes, xh, h_ids, config.hlen, 'h_nn',
                      dtype=config.bf_dtype)
    o_en = element_nn(config.o_nodes, xo, o_ids, config.olen, 'o_nn',
                      dtype=config.bf_dtype)
    nn_en = tf.add(h_en, o_en)
    energy_cost = tf.reduce_mean(tf.math.squared_difference(nn_en,
                                                     tf.reshape(y, (-1,1))),
                          name='energy_cost')
    squared_error = tf.math.squared_difference(nn_en, tf.reshape(y, (-1,1)))
    difference = tf.subtract(nn_en, tf.reshape(y, (-1,1)))

    # The gradients of the neural network WRT the basis functions
    dnn_dh, dnn_do = tf.gradients(nn_en, [xh, xo])[0:2]

    # Tensor contraction to [basis_size, ngrum_atoms, 3]
    h_bas_cart_grads = tf.einsum('ijkl,ij->ikl', h_bas_grads, dnn_dh)
    # Here we go to [batch_size, num_atoms, 3]
    h_cart_grads = tf.math.segment_sum(h_bas_cart_grads, h_ids)

    o_bas_cart_grads = tf.einsum('ijkl,ij->ikl', o_bas_grads, dnn_do)
    o_cart_grads = tf.math.segment_sum(o_bas_cart_grads, o_ids)

    # This gives us the total correction gradient
    corr_grad = tf.add(h_cart_grads, o_cart_grads)
    grad_error = tf.math.squared_difference(corr_grad, ref_grads, name='grad_error')
    #
    # This is replaced by MSE above
    #ge
    # This gives us the error in gradient
#    grad_error = tf.subtract(corr_grad, ref_grads, name='grad_error')
    # We need the norm of the error in gradient along the axis of xyz
#    grad_norm = tf.norm(grad_error, ord='euclidean', axis=2, name='grad_norm')




    # Sum before reduce mean, because otherwise the 0 padded axes will
    # affect the meanc
    cart_sum = tf.reduce_sum(grad_error, axis=2, name='cart_grad_sum')
    geom_sum = tf.reduce_sum(grad_error, axis=1, name='geom_grad_sum')
    geom_mean = tf.reduce_mean(geom_sum, name='grad_geom_mean')
    grad_cost = tf.multiply(geom_mean, grad_scale, name='grad_cost')
    cost = tf.add(grad_cost, energy_cost, name='total_cost')

    # Optimizer
    optimizer = define_optimizer(config)

    # Training and statistics
    train_step = optimizer.minimize(energy_cost)
    train_grads = optimizer.minimize(grad_cost)
    train_all = optimizer.minimize(cost)

    # Saving info
    saver = tf.train.Saver(max_to_keep=100000)

    return AttrDict(locals())


def define_oh_builder(graph, config, sess):
    """
    This subroutine defines a builder using the tf.SavedModel API in order to
    create a tensorflow serving model for molecular dynamics purposes.

    Parameters
    ----------
    graph: AttrDict
        Contains the TF graph
    config: AttrDict
        Contains the configuration option config.export_path for the folder
        for the saved model.
    sess: tf.Session

    Returns
    -------
    builder: tf.saved_model.builder.SavedModelBuilder

    Notes
    -----
    This is hardcoded to use specifically named variables from the graph input.

    This is also heavily inspired by the tensorflow SavedModel API tutorial.

    """
    builder = tf.saved_model.builder.SavedModelBuilder(config.export_path)
    tensor_info_xh = tf.saved_model.utils.build_tensor_info(graph.xh)
    tensor_info_xo = tf.saved_model.utils.build_tensor_info(graph.xo)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(graph.nn_en)
    tensor_info_h_ids = tf.saved_model.utils.build_tensor_info(graph.h_ids)
    tensor_info_o_ids = tf.saved_model.utils.build_tensor_info(graph.o_ids)
    tensor_info_h_grad = tf.saved_model.utils.build_tensor_info(graph.dnn_dh)
    tensor_info_o_grad = tf.saved_model.utils.build_tensor_info(graph.dnn_do)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'h_basis': tensor_info_xh,
                    'o_basis': tensor_info_xo,
                    'h_bas2mol': tensor_info_h_ids,
                    'o_bas2mol': tensor_info_o_ids},
            outputs={'correction_energies': tensor_info_y,
                     'h_basis_grad': tensor_info_h_grad,
                     'o_basis_grad': tensor_info_o_grad},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        },
    )
    return builder


def define_optimizer(config):
    """
    Return an optimizer based on the info

    Parameters
    ----------
    config: AttrDict
        Dict-like object with keywords for the optimizer

    Returns
    -------
    tf.train.Optimizer

    Notes
    -----

    Config must contain keywords:
    optimizer: 'adam'
    """
    lr = config['learning_rate']
    if config['optimizer'] == 'adam':
        return tf.train.AdamOptimizer()
    elif config['optimizer'] == 'gradient_descent':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    else:
        raise TypeError


def train_model(config, graph, train_dict, valid_dict=None):
    """
    Train the neural network for a designated amount of epochs

    Parameters
    ----------
    config: AttrDict
        Configuration options
    graph: AttrDict
        TF Graph to execute

    Returns
    -------
    None

    Notes
    -----
    Uses the feed-dict method of input.

    """
    start_epoch = config['start_epoch']
    end_epoch = start_epoch + config['epochs']
    train_scores = []
    valid_scores = []
    epoch_scores = []
    save_name = None

    # Main training loop
    with tf.Session() as sess:
        if config['restore_path']:
            graph.saver.restore(sess, config['restore_path'])
        else:
            sess.run(tf.global_variables_initializer())
        print("Initial train cost:", config['cost'].eval(feed_dict=train_dict))
        for epoch in range(start_epoch, end_epoch + 1):
            sess.run(config['train'], feed_dict=train_dict)

            if epoch % config['score_freq'] == 0:
                train_scores.append(config['cost'].eval(feed_dict=train_dict))
                if valid_dict:
                    valid_scores.append(config['cost'].eval(feed_dict=valid_dict))
                epoch_scores.append(epoch)
            if epoch % config['print_freq'] == 0:
                if valid_dict:
                    print(epoch_scores[-1], train_scores[-1], valid_scores[-1])
                else:
                    print(epoch_scores[-1], train_scores[-1])
            if epoch % config['save_freq'] == 0:
                save_name = config.save_name.format(epoch)
                graph.saver.save(sess, config.save_name.format(epoch))
                if config['score_path'] and valid_dict:
                    pickle.dump(valid_scores, open(config['score_path'].format(epoch),
                                             'wb'))
        if config.builder:
            builder = config.builder(graph, config, sess)
            builder.save()
        config['start_epoch'] = end_epoch
        config['restore_path'] = save_name
