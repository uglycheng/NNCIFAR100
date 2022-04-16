import tensorflow as tf

class LinearRegression(tf.keras.Model):
    # This model is just for testing whether the training and predicting process can work normally.
    def __init__(self,seed,**kwargs):
        super().__init__(**kwargs)
        initializer = tf.random_normal_initializer(seed=29)
        self.weightes = tf.Variable(initializer(shape=[32*32*3,100],dtype=tf.float32))

    def call(self,inputs):
        inp = tf.reshape(inputs,shape=[inputs.shape[0],-1])
        out = tf.matmul(inp,self.weightes)
        return out



class SuperSimpleCNN(tf.keras.Model):
    # This model is just for testing whether the training and predicting process can work normally.
    # Its architecture is the same as the CNN in Chapter 5 of Deep Learning with Python by Chollet
    def __init__(self,seed, **kwargs):
        super().__init__(**kwargs)
        initializer = tf.random_normal_initializer(seed=seed)
        # self.conv_kernel1 = tf.Variable(initializer(shape=[3, 3, 3, 32],dtype=tf.float32))
        # self.conv_kernel2 = tf.Variable(initializer(shape=[3, 3, 32, 64],dtype=tf.float32))
        # self.conv_kernel3 = tf.Variable(initializer(shape=[3, 3, 64, 64], dtype=tf.float32))
        # self.dense1 = tf.Variable(initializer(shape=[32*32*64, 64], dtype=tf.float32))
        # self.dense2 = tf.Variable(initializer(shape=[64,100],dtype=tf.float32))
        # self.bias1 = tf.Variable(initializer(shape=[64,],dtype=tf.float32))
        # self.bias2 = tf.Variable(initializer(shape=[100,],dtype=tf.float32))

        self.conv1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3),kernel_initializer=initializer,bias_initializer='zeros')
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2,2))
        self.conv2 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer=initializer,bias_initializer='zeros')
        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializer,bias_initializer='zeros')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64,activation='relu',kernel_initializer=initializer,bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(100,kernel_initializer=initializer,bias_initializer='zeros')

    def call(self,inputs):
        # temp = tf.nn.conv2d(inputs,self.conv_kernel1,strides=(1,1),padding='valid',data_format=None)

        temp = self.conv1(inputs)
        temp = self.max_pool1(temp)
        temp = self.conv2(temp)
        temp = self.max_pool2(temp)
        temp = self.conv3(temp)
        temp = self.flatten(temp)
        temp = self.dense1(temp)
        temp = self.dense2(temp)
        return temp
