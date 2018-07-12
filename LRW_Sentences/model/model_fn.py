"""Define the model."""

import tensorflow as tf
import was # stands for watch-attend-spell related code
import logging
import utils



def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """


    
    images = inputs['images']
    source_sequence_length = inputs['source_sequence_length']
    
    decoder_inputs = None
    targets = None
    target_sequence_length = None
    
    if is_training:
        decoder_inputs = inputs['targets_inputs']
        targets = inputs['targets_outputs']
        target_sequence_length = inputs['target_sequence_length']
        
    out = images
    out = tf.reshape(out, shape= [-1, 112,112,5]) # reshaping into [batch_size*sequence_length,112,112,5]

    # define CNN feature extractor
    logging.info('Building watcher')
    with tf.variable_scope('CNN_features'):
            out = tf.layers.conv2d(images, filters = 96, kernel_size = 3, padding='valid', name='conv1_lip')
            out = tf.layers.batch_normalization(out, training= is_training, name= 'bn1_lip')
            out = tf.nn.relu(out, name= 'relu1_lip')
            out = tf.layers.max_pooling2d(out, pool_size = 3, strides= 2, name = 'pool1_lip')
            
            
            out = tf.layers.conv2d(out, filters = 256, kernel_size = 5, padding='valid', name='conv2_lip')
            out = tf.layers.batch_normalization(out, training= is_training, name= 'bn2_lip')
            out = tf.nn.relu(out, name= 'relu2_lip')
            out = tf.layers.max_pooling2d(out, pool_size = 3, strides= 2, name = 'pool2_lip')
            
            
            out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, padding='valid', name='conv3_lip')
            out = tf.layers.batch_normalization(out, training= is_training, name= 'bn3_lip')
            out = tf.nn.relu(out, name= 'relu3_lip')
            
            
            out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, padding='valid', name='conv4_lip')
            out = tf.layers.batch_normalization(out, training= is_training, name= 'bn4_lip')
            out = tf.nn.relu(out, name= 'relu4_lip')
            
            
            out = tf.layers.conv2d(out, filters = 96, kernel_size = 3, padding='valid', name='conv5_lip')
            out = tf.layers.batch_normalization(out, training= is_training, name= 'bn5_lip')
            out = tf.nn.relu(out, name= 'relu5_lip')
            out = tf.layers.max_pooling2d(out, pool_size = 3, strides= 3, name = 'pool5_lip')
            
            # fc6_lip
            out = tf.layers.flatten(out, name = 'flatten_lip')
            out = tf.layers.dense(out, units = 512, name = 'fc6_lip')    # (None, 512)
            
    # shape defined as per demo data, need to change after
    out = tf.reshape(out, shape = [params.batch_size,-1,512]) # this will reshape a tensor back to batch_size as first dimension.
    
    forward_cell_list = []
    for layer in range(params.num_vis_enc_layer):
            with tf.variable_scope('fw_cell_{}'.format(layer)):
                cell = was.ops.lstm_cell(params.num_vis_enc_units, params.vis_enc_dropout, is_training)

            forward_cell_list.append(cell)
            
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(forward_cell_list)
    
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=out,
                                   dtype=tf.float32)
        
    logging.info('Building speller')
    with tf.variable_scope('speller'):
        decoder_outputs, final_context_state, final_sequence_length = was.model.speller(
            encoder_outputs, encoder_state, decoder_inputs,
            source_sequence_length, target_sequence_length,
            is_training, params['decoder'])
    
    with tf.name_scope('prediction'):
        if not is_training and params['decode']['beam_width'] > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            sample_ids = tf.to_int32(tf.argmax(logits, -1))
            
    with tf.name_scope('cross_entropy'):
        loss = was.ops.compute_loss(
            logits, targets, final_sequence_length, target_sequence_length, is_training)

    with tf.name_scope('metrics'):
        edit_distance = was.ops.edit_distance(
            sample_ids, targets, utils.EOS_ID, params.mapping)
        
    return sample_ids, loss, edit_distance


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        sample_ids, loss, edit_distance = build_model(is_training, inputs, params)
    
    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'edit_distance': tf.metrics.mean(edit_distance),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('edit_distance', edit_distance)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = sample_ids
    model_spec['loss'] = loss
    model_spec['edit_distance'] = edit_distance
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
