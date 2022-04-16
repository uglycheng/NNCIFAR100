import tensorflow as tf


def cast2float_cifar100(x):
    image = tf.cast(x['image'], dtype=tf.float32)
    label = tf.squeeze(tf.one_hot(x['label'], depth=100, on_value=1.0, off_value=0.0))
    return image, label

def make_train_batches_no_shuffle_cifar100(ds,train_batch_size):
    return (
      ds
      .cache()
      .batch(train_batch_size)
      .map(cast2float_cifar100, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))

def make_train_batches_cifar100(ds,shuffle_buffer_size,shuffle_seed,train_batch_size):
    return (
      ds
      .cache()
      .shuffle(shuffle_buffer_size,seed=shuffle_seed)
      .batch(train_batch_size)
      .map(cast2float_cifar100, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))

def make_test_batches_cifar100(ds,test_batch_size):
    return(
     ds
     .cache()
     .batch(test_batch_size)
     .map(cast2float_cifar100, num_parallel_calls=tf.data.AUTOTUNE)
     .prefetch(tf.data.AUTOTUNE))
