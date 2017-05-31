import tensorflow as tf

# DISCLAIMER:
# Parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package
def masked_logit_cross_entropy(preds, labels, mask):
    """Logit cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_sum(loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(loss)


def masked_l2(preds, actuals, mask):
    """L2 loss with masking."""
    loss = tf.nn.l2(preds, actuals)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
