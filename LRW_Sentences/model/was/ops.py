import tensorflow as tf
from .. import utils

__all__ = [
    'lstm_cell',
    'bilstm',
    'pyramidal_bilstm',
    'dense_to_sparse',
    'edit_distance'

]


def lstm_cell(num_units, dropout, mode):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if dropout > 0.0:
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=cell, input_keep_prob=(1.0 - dropout))

    return cell


def bilstm(inputs,
           sequence_length,
           num_units,
           dropout,
           mode):

    with tf.variable_scope('fw_cell'):
        forward_cell = lstm_cell(num_units, dropout, mode)
    with tf.variable_scope('bw_cell'):
        backward_cell = lstm_cell(num_units, dropout, mode)

    return tf.nn.bidirectional_dynamic_rnn(
        forward_cell,
        backward_cell,
        inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)


def pyramidal_stack(outputs, sequence_length):
    shape = tf.shape(outputs)
    batch_size, max_time = shape[0], shape[1]
    num_units = outputs.get_shape().as_list()[-1]
    paddings = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
    outputs = tf.pad(outputs, paddings)

    '''
    even_time = outputs[:, ::2, :]
    odd_time = outputs[:, 1::2, :]

    concat_outputs = tf.concat([even_time, odd_time], -1)
    '''

    concat_outputs = tf.reshape(outputs, (batch_size, -1, num_units * 2))

    return concat_outputs, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)


def pyramidal_bilstm(inputs,
                     sequence_length,
                     mode,
                     hparams):

    outputs = inputs

    for layer in range(hparams.num_layers):
        with tf.variable_scope('bilstm_{}'.format(layer)):
            outputs, state = bilstm(
                outputs, sequence_length, hparams.num_units, hparams.dropout, mode)

            outputs = tf.concat(outputs, -1)

            if layer != 0:
                outputs, sequence_length = pyramidal_stack(
                    outputs, sequence_length)

    return (outputs, sequence_length), state


def compute_loss(logits, targets, final_sequence_length, target_sequence_length, is_training):

    assert is_training

    if is_training:
        target_weights = tf.sequence_mask(
            target_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
    else:
        '''
        # Reference: https://github.com/tensorflow/nmt/issues/2
        # Note that this method always trim the tensor with larger length to shorter one, 
        # and I think it is unfair. 
        # Consider targets = [[3, 3, 2]], and logits with shape [1, 2, VOCAB_SIZE]. 
        # This method will trim targets to [[3, 3]] and compute sequence_loss on new targets and logits.
        # However, I think the truth is that the model predicts less word than ground truth does,
        # and hence, both targets and logits should be padded to the same sequence length (dimension 1)
        # to compute loss.
        current_sequence_length = tf.to_int32(
            tf.minimum(tf.shape(targets)[1], tf.shape(logits)[1]))
        targets = tf.slice(targets, begin=[0, 0],
                           size=[-1, current_sequence_length])
        logits = tf.slice(logits, begin=[0, 0, 0],
                          size=[-1, current_sequence_length, -1])
        target_weights = tf.sequence_mask(
            target_sequence_length, maxlen=current_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)
        '''

        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(
            tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])]], constant_values=utils.EOS_ID)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max(
            [target_sequence_length, final_sequence_length], 0)

        target_weights = tf.sequence_mask(
            sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets, target_weights)

    return loss

def dense_to_sparse(tensor, eos_id, merge_repeated=True):
    if merge_repeated:
        added_values = tf.cast(
            tf.fill((tf.shape(tensor)[0], 1), eos_id), tensor.dtype)

        # merge consecutive values
        concat_tensor = tf.concat((tensor, added_values), axis=-1)
        diff = tf.cast(concat_tensor[:, 1:] - concat_tensor[:, :-1], tf.bool)

        # trim after first eos token
        eos_indices = tf.where(tf.equal(concat_tensor, eos_id))
        first_eos = tf.segment_min(eos_indices[:, 1], eos_indices[:, 0])
        mask = tf.sequence_mask(first_eos, maxlen=tf.shape(tensor)[1])

        indices = tf.where(diff & mask & tf.not_equal(tensor, -1))
        values = tf.gather_nd(tensor, indices)
        shape = tf.shape(tensor, out_type=tf.int64)

        return tf.SparseTensor(indices, values, shape)
    else:
        return tf.contrib.layers.dense_to_sparse(tensor, eos_id)


def edit_distance(hypothesis, truth, eos_id, mapping=None): # calculates levenshtein distance

    if mapping:
        mapping = tf.convert_to_tensor(mapping)
        hypothesis = tf.nn.embedding_lookup(mapping, hypothesis)
        truth = tf.nn.embedding_lookup(mapping, truth)

    hypothesis = dense_to_sparse(hypothesis, eos_id, merge_repeated=True)
    truth = dense_to_sparse(truth, eos_id, merge_repeated=True)

    return tf.edit_distance(hypothesis, truth, normalize=True)