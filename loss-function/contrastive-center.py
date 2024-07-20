centers = tf.Variable(tf.zeros([16, 128]), trainable=False)

def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, 'float32')
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def center_loss(y_true, y_pred):
    alpha = 0.5
    y_true = tf.cast(y_true, 'int32')
    y_true_matrix = tf.one_hot(y_true, depth=16)
    centers_batch = tf.gather(centers, y_true)
    diff = centers_batch - y_pred
    unique_labels, unique_idx, unique_counts = tf.unique_with_counts(y_true)
    appear_times = tf.gather(unique_counts, unique_idx)
    appear_times = tf.reshape(appear_times, (-1, 1))
    diff /= tf.cast((1 + appear_times), tf.float32)
    diff *= alpha
    centers_update = tf.tensor_scatter_nd_sub(centers, tf.reshape(y_true, [-1, 1]), diff)
    with tf.control_dependencies([centers_update]):
        centers_batch_updated = tf.gather(centers, y_true)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - centers_batch_updated), axis=1))
    return loss

def combined_loss(y_true, y_pred):
    tf.print("y_true shape:", tf.shape(y_true))
    tf.print("y_pred shape:", tf.shape(y_pred))

    # Ensure y_true has the expected shape
    if tf.shape(y_true)[1] != 3:
        tf.print("Error: y_true does not have the expected shape.")
        return 0.0

    # Ensure y_pred has the expected shape (batch_size, 257)
    if tf.shape(y_pred)[1] != 257:
        tf.print("Error: y_pred does not have the expected shape.")
        return 0.0

    # Unpack y_pred
    distance = y_pred[:, 0]
    embedding_1 = y_pred[:, 1:129]
    embedding_2 = y_pred[:, 129:]

    # Calculate losses
    cont_loss = contrastive_loss(y_true[:, 0], distance)
    cent_loss_1 = center_loss(tf.cast(y_true[:, 1], tf.int32), embedding_1)
    cent_loss_2 = center_loss(tf.cast(y_true[:, 2], tf.int32), embedding_2)

    total_loss = cont_loss + 0.1 * (cent_loss_1 + cent_loss_2)
    tf.print("Total combined loss:", total_loss)
    return total_loss
