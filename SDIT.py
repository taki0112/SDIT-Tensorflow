from ops import *
from utils import *
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
from glob import glob
from tqdm import tqdm

class SDIT() :
    def __init__(self, sess, args):
        self.model_name = 'SDIT'
        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.dataset_path = os.path.join('./dataset', self.dataset_name)
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type
        self.attention = args.attention

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch

        if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
            self.label_list = args.label_list
        else :
            self.dataset_path = os.path.join(self.dataset_path, 'train')
            self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]


        self.c_dim = len(self.label_list)

        """ Weight """
        self.adv_weight = args.adv_weight
        self.rec_weight = args.rec_weight
        self.cls_weight = args.cls_weight
        self.noise_weight = args.noise_weight
        self.gp_weight = args.gp_weight

        self.sn = args.sn

        """ Generator """
        self.n_res = args.n_res
        self.style_dim = args.style_dim
        self.num_style = args.num_style

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_ch = args.img_ch

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# selected_attrs : ", self.label_list)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# attention : ", self.attention)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, c, style, reuse=False, scope="generator"):
        channel = self.ch
        c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        c = tf.tile(c, [1, x_init.shape[1], x_init.shape[2], 1])
        x = tf.concat([x_init, c], axis=-1)

        with tf.variable_scope(scope, reuse=reuse) :
            """ Encoder """
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='conv_'+str(i))
                x = instance_norm(x, scope='down_ins_norm_'+str(i))
                x = relu(x)

                channel = channel * 2

            """ Bottleneck """
            # Encoder Bottleneck
            for i in range(self.n_res) :
                x = resblock(x, channel, use_bias=False, sn=self.sn, scope='encoder_resblock_' + str(i))

            attention = x
            adaptive = x

            # Adaptive Bottleneck
            mu, var = self.MLP(style, channel)
            for i in range(self.n_res - 2) :
                idx = 2 * i
                adaptive = adaptive_resblock(adaptive, channel, mu[idx], var[idx], mu[idx + 1], var[idx + 1], use_bias=True, sn=self.sn, scope='ada_resbloack_' + str(i))

            if self.attention :
                # Attention Bottleneck
                for i in range(self.n_res - 1) :
                    attention = resblock(attention, channel, use_bias=False, sn=self.sn, scope='attention_resblock_' + str(i))

                attention = conv(attention, 1, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='attention_conv')
                attention = instance_norm(attention, scope='attention_ins_norm')
                attention = sigmoid(attention)

                x = attention * adaptive

                # attention_map = tf.concat([attention, attention, attention], axis=-1) * 2 - 1
                # attention_map = up_sample(attention_map, scale_factor=4)

            else :
                x = adaptive

            """ Decoder """
            # Up-Sampling
            for i in range(2):
                x = deconv(x, channel // 2, kernel=4, stride=2, use_bias=False, sn=self.sn, scope='deconv_' + str(i))
                x = instance_norm(x, scope='up_ins_norm' + str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, sn=self.sn, scope='G_logit')
            x = tanh(x)

            return x

    def MLP(self, style, channel, scope='MLP'):
        with tf.variable_scope(scope):
            x = style

            for i in range(2):
                x = fully_connected(x, channel, sn=self.sn, scope='FC_' + str(i))
                x = relu(x)

            mu_list = []
            var_list = []

            for i in range(8):
                mu = fully_connected(x, channel, sn=self.sn, scope='FC_mu_' + str(i))
                var = fully_connected(x, channel, sn=self.sn, scope='FC_var_' + str(i))

                mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
                var = tf.reshape(var, shape=[-1, 1, 1, channel])

                mu_list.append(mu)
                var_list.append(var)

            return mu_list, var_list

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        with tf.variable_scope(scope, reuse=reuse) :
            channel = self.ch
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', use_bias=True, sn=self.sn, scope='conv_' + str(i))
                x = lrelu(x, 0.01)

                channel = channel * 2

            c_kernel = int(self.img_height / np.power(2, self.n_dis))

            logit = conv(x, channels=1, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='D_logit')

            c = conv(x, channels=self.c_dim, kernel=c_kernel, stride=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='D_label')
            c = tf.reshape(c, shape=[-1, self.c_dim])

            noise = conv(x, channels=self.style_dim, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=self.sn, scope='D_noise')
            noise = fully_connected(noise, units=self.style_dim, use_bias=True, sn=self.sn, scope='fc_0')
            noise = relu(noise)
            noise = fully_connected(noise, units=self.style_dim, use_bias=True, sn=self.sn, scope='fc_1')

            return logit, c, noise

    ##################################################################################
    # Model
    ##################################################################################

    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit, _, _ = self.discriminator(interpolated, reuse=True, scope=scope)


        GP = 0

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=-1) # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = self.gp_weight * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.gp_weight * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def build_model(self):
        label_fix_onehot_list = []

        """ Input Image"""
        if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
            img_class = ImageData_celebA(self.img_height, self.img_width, self.img_ch, self.dataset_path,
                                         self.label_list, self.augment_flag)
            img_class.preprocess(self.phase)

        else:
            img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.label_list,
                                   self.augment_flag)
            img_class.preprocess()

            label_fix_onehot_list = img_class.label_onehot_list
            label_fix_onehot_list = tf.tile(tf.expand_dims(label_fix_onehot_list, axis=1), [1, self.batch_size, 1])

        dataset_num = len(img_class.image)
        print("Dataset number : ", dataset_num)

        if self.phase == 'train' :
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

            if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
                img_and_label = tf.data.Dataset.from_tensor_slices(
                    (img_class.image, img_class.label, img_class.train_label_onehot_list))
            else:
                img_and_label = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.label))

            gpu_device = '/gpu:0'
            img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
                map_and_batch(img_class.image_processing, self.batch_size, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            img_and_label_iterator = img_and_label.make_one_shot_iterator()

            if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
                self.x_real, label_org, label_fix_onehot_list = img_and_label_iterator.get_next()
                label_trg = tf.random_shuffle(label_org)  # Target domain labels
                label_fix_onehot_list = tf.transpose(label_fix_onehot_list, perm=[1, 0, 2])
            else:
                self.x_real, label_org = img_and_label_iterator.get_next()
                label_trg = tf.random_shuffle(label_org)  # Target domain labels


            """ Define Generator, Discriminator """
            fake_style_code = tf.random_normal(shape=[self.batch_size, self.style_dim])
            x_fake = self.generator(self.x_real, label_trg, fake_style_code) # real a

            recon_style_code = tf.random_normal(shape=[self.batch_size, self.style_dim])
            x_recon = self.generator(x_fake, label_org, recon_style_code, reuse=True) # real b

            real_logit, real_cls, _ = self.discriminator(self.x_real)
            fake_logit, fake_cls, fake_noise = self.discriminator(x_fake, reuse=True)


            """ Define Loss """
            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
                GP = self.gradient_panalty(real=self.x_real, fake=x_fake)
            else :
                GP = 0

            g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit)
            g_cls_loss = self.cls_weight * classification_loss(logit=fake_cls, label=label_trg)
            g_rec_loss = self.rec_weight * L1_loss(self.x_real, x_recon)
            g_noise_loss = self.noise_weight * L1_loss(fake_style_code, fake_noise)

            d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit) + GP
            d_cls_loss = self.cls_weight * classification_loss(logit=real_cls, label=label_org)
            d_noise_loss = self.noise_weight * L1_loss(fake_style_code, fake_noise)

            self.d_loss = d_adv_loss + d_cls_loss + d_noise_loss
            self.g_loss = g_adv_loss + g_cls_loss + g_rec_loss + g_noise_loss


            """ Result Image """
            if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
                self.x_fake_list = []

                for _ in range(self.num_style):
                    random_style_code = tf.random_normal(shape=[self.batch_size, self.style_dim])
                    self.x_fake_list.append(tf.map_fn(lambda c : self.generator(self.x_real, c, random_style_code, reuse=True), label_fix_onehot_list, dtype=tf.float32))

            else :
                self.x_fake_list = []

                for _ in range(self.num_style) :
                    random_style_code = tf.random_normal(shape=[self.batch_size, self.style_dim])
                    self.x_fake_list.append(tf.map_fn(lambda c : self.generator(self.x_real, c, random_style_code, reuse=True), label_fix_onehot_list, dtype=tf.float32))



            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            self.g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=G_vars)
            self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=D_vars)


            """" Summary """
            self.Generator_loss = tf.summary.scalar("g_loss", self.g_loss)
            self.Discriminator_loss = tf.summary.scalar("d_loss", self.d_loss)

            self.g_adv_loss = tf.summary.scalar("g_adv_loss", g_adv_loss)
            self.g_cls_loss = tf.summary.scalar("g_cls_loss", g_cls_loss)
            self.g_rec_loss = tf.summary.scalar("g_rec_loss", g_rec_loss)
            self.g_noise_loss = tf.summary.scalar("g_noise_loss", g_noise_loss)

            self.d_adv_loss = tf.summary.scalar("d_adv_loss", d_adv_loss)
            self.d_cls_loss = tf.summary.scalar("d_cls_loss", d_cls_loss)
            self.d_noise_loss = tf.summary.scalar("d_noise_loss", d_noise_loss)

            self.g_summary_loss = tf.summary.merge([self.Generator_loss, self.g_adv_loss, self.g_cls_loss, self.g_rec_loss, self.g_noise_loss])
            self.d_summary_loss = tf.summary.merge([self.Discriminator_loss, self.d_adv_loss, self.d_cls_loss, self.d_noise_loss])

        else :
            """ Test """
            if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
                img_and_label = tf.data.Dataset.from_tensor_slices(
                    (img_class.test_image, img_class.test_label, img_class.test_label_onehot_list))
                dataset_num = len(img_class.test_image)

                gpu_device = '/gpu:0'
                img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
                    map_and_batch(img_class.image_processing, batch_size=self.batch_size, num_parallel_batches=16,
                                  drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

                img_and_label_iterator = img_and_label.make_one_shot_iterator()

                self.x_test, _, self.test_label_fix_onehot_list = img_and_label_iterator.get_next()
                self.test_img_placeholder = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch])
                self.test_label_fix_placeholder = tf.placeholder(tf.float32, [self.c_dim, 1, self.c_dim])

                self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')  # Custom Image
                custom_label_fix_onehot_list = tf.transpose(np.expand_dims(label2onehot(self.label_list), axis=0), perm=[1, 0, 2]) # [c_dim, bs, c_dim]

                """ Test Image """
                test_random_style_code = tf.random_normal(shape=[1, self.style_dim])

                self.x_test_fake_list = tf.map_fn(lambda c : self.generator(self.test_img_placeholder, c, test_random_style_code), self.test_label_fix_placeholder, dtype=tf.float32)
                self.custom_fake_image = tf.map_fn(lambda c : self.generator(self.custom_image, c, test_random_style_code, reuse=True), custom_label_fix_onehot_list, dtype=tf.float32)

            else :
                self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')  # Custom Image
                custom_label_fix_onehot_list = tf.transpose(np.expand_dims(label2onehot(self.label_list), axis=0), perm=[1, 0, 2]) # [c_dim, bs, c_dim]

                test_random_style_code = tf.random_normal(shape=[1, self.style_dim])
                self.custom_fake_image = tf.map_fn(lambda c : self.generator(self.custom_image, c, test_random_style_code), custom_label_fix_onehot_list, dtype=tf.float32)



    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag :
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay

            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {
                    self.lr : lr
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.d_optimizer, self.d_loss, self.d_summary_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                g_loss = None
                if (counter - 1) % self.n_critic == 0 :
                    real_images, fake_images, _, g_loss, summary_str = self.sess.run([self.x_real, self.x_fake_list, self.g_optimizer, self.g_loss, self.g_summary_loss], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss

                print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :
                    real_image = np.expand_dims(real_images[0], axis=0)
                    save_images(real_image, [1, 1],
                                './{}/real_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))

                    merge_fake_x = None

                    for ns in range(self.num_style) :
                        fake_img = np.transpose(fake_images[ns], axes=[1, 0, 2, 3, 4])[0]

                        if ns == 0 :
                            merge_fake_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                        else :
                            x = return_images(fake_img, [1, self.c_dim])
                            merge_fake_x = np.concatenate([merge_fake_x, x], axis=0)

                    merge_fake_x = np.expand_dims(merge_fake_x, axis=0)
                    save_images(merge_fake_x, [1, 1],
                                './{}/fake_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx+1))

                if np.mod(counter - 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        if self.attention:
            attention = '_attention'
        else:
            attention = ''

        return "{}_{}_{}_{}adv_{}rec_{}cls_{}noise{}{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                               self.adv_weight, self.rec_weight, self.cls_weight, self.noise_weight,
                                                               sn, attention)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_files = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'test')) + glob('./dataset/{}/{}/*.png'.format(self.dataset_name, 'test'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        custom_image_folder = os.path.join(self.result_dir, 'custom_fake_images')
        check_folder(custom_image_folder)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        # Custom Image
        for sample_file in tqdm(test_files):
            print("Processing image: " + sample_file)
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(custom_image_folder, '{}'.format(os.path.basename(sample_file)))

            merge_x = None

            for i in range(self.num_style) :
                fake_img = self.sess.run(self.custom_fake_image, feed_dict={self.custom_image: sample_image})
                fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]

                if i == 0:
                    merge_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                else :
                    x = return_images(fake_img, [1, self.c_dim])
                    merge_x = np.concatenate([merge_x, x], axis=0)

            merge_x = np.expand_dims(merge_x, axis=0)

            save_images(merge_x, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_width * self.c_dim, self.img_height * self.num_style))
            index.write("</tr>")

        if self.dataset_name == 'celebA-HQ' or self.dataset_name == 'celebA':
            # CelebA
            celebA_image_folder = os.path.join(self.result_dir, 'celebA_real_fake_images')
            check_folder(celebA_image_folder)
            real_images, real_label_fixes = self.sess.run([self.x_test, self.test_label_fix_onehot_list])

            for i in tqdm(range(len(real_images))) :

                real_path = os.path.join(celebA_image_folder, 'real_{}.png'.format(i))
                fake_path = os.path.join(celebA_image_folder, 'fake_{}.png'.format(i))

                real_img = np.expand_dims(real_images[i], axis=0)
                real_label_fix = np.expand_dims(real_label_fixes[i], axis=1)

                merge_x = None

                for ns in range(self.num_style) :
                    fake_img = self.sess.run(self.x_test_fake_list, feed_dict={self.test_img_placeholder: real_img, self.test_label_fix_placeholder:real_label_fix})
                    fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]

                    if ns == 0:
                        merge_x = return_images(fake_img, [1, self.c_dim])  # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                    else:
                        x = return_images(fake_img, [1, self.c_dim])
                        merge_x = np.concatenate([merge_x, x], axis=0)

                merge_x = np.expand_dims(merge_x, axis=0)

                save_images(real_img, [1, 1], real_path)
                save_images(merge_x, [1, 1], fake_path)

                index.write("<td>%s</td>" % os.path.basename(real_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (real_path if os.path.isabs(real_path) else (
                    '../..' + os.path.sep + real_path), self.img_width, self.img_height))

                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (fake_path if os.path.isabs(fake_path) else (
                    '../..' + os.path.sep + fake_path), self.img_width * self.c_dim, self.img_height * self.num_style))
                index.write("</tr>")

        index.close()