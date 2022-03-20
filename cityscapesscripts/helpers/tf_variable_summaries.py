import tensorflow as tf


def add_variable_summaries(variable, scope):
    """
    Attache certains récapitulatifs à un tenseur pour la visualisation sur
    TensorBoard, à savoir la moyenne, l'écart type, le minimum, le maximum et
    l'histogramme.

    Arguments :
      var (Variable TensorFlow) : Une variable TensorFlow de n'importe quelle
          forme à laquelle ajouter des opérations de résumé. Doit être un type
          de données numériques.
    """
    with tf.name_scope(scope):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(variable))
        tf.summary.scalar('min', tf.reduce_min(variable))
        tf.summary.histogram('histogram', variable)
