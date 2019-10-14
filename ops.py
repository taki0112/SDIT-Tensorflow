import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_regularizer = None
weight_regularizer_fully = None


##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)

        return x + x_init

def adaptive_resblock(x_init, channels, gamma1, beta1, gamma2, beta2, use_bias=True, sn=False, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, gamma1, beta1)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, gamma2, beta2)

        return x + x_init


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x):
    return tf.sigmoid(x)

##################################################################################
# Pooling & Resize
##################################################################################

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_bilinear(x, size=new_size)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real_logit, fake_logit):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real_logit, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake_logit))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake_logit):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake_logit)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake_logit, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake_logit)

    loss = fake_loss

    return loss

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

    return loss

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss