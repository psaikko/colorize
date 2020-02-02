import tensorflow as tf

# Residual block as in
# https://web.eecs.umich.edu/~justincj/papers/eccv16/JohnsonECCV16Supplementary.pdf
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters

        self.layers = [
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.activations.relu,
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3)),
            tf.keras.layers.BatchNormalization()
        ]
        for (i, layer) in enumerate(self.layers):
             self.__setattr__("layer_%d"%i, layer)

    def call(self, x):
        # Residual: add input to output
        res = tf.keras.layers.Cropping2D(((2,2),(2,2)))(x)
        for layer in self.layers:
            x = layer(x)
        return tf.keras.backend.sum([x, res], axis=0)

class Colorizer(tf.keras.Model):
    def __init__(self, scale=32, out_channels=3):
        super(Colorizer, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(filters=scale, kernel_size=(9,9), strides=1, padding="same")
        self.norm_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.activations.relu

        self.conv_2 = tf.keras.layers.Conv2D(filters=scale*2, kernel_size=(3,3), strides=2, padding="same")
        self.norm_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.activations.relu

        self.conv_3 = tf.keras.layers.Conv2D(filters=scale*4, kernel_size=(3,3), strides=2, padding="same")
        self.norm_3 = tf.keras.layers.BatchNormalization()
        self.relu_3 = tf.keras.activations.relu

        self.fusion_conv = tf.keras.layers.Conv2D(filters=scale*4, kernel_size=(1,1), strides=1)

        self.res_block_1 = ResidualBlock(scale*4)
        self.res_block_2 = ResidualBlock(scale*4)
        self.res_block_3 = ResidualBlock(scale*4)
        self.res_block_4 = ResidualBlock(scale*4)
        self.res_block_5 = ResidualBlock(scale*4)

        # equivalent to "fractionally strided convolutions"
        self.conv_4 = tf.keras.layers.Conv2DTranspose(filters=scale*2, kernel_size=(3,3), strides=2, padding="same")
        self.norm_4 = tf.keras.layers.BatchNormalization()
        self.relu_4 = tf.keras.activations.relu

        self.conv_5 = tf.keras.layers.Conv2DTranspose(filters=scale, kernel_size=(3,3), strides=2, padding="same")
        self.norm_5 = tf.keras.layers.BatchNormalization()
        self.relu_5 = tf.keras.activations.relu

        self.conv_6 = tf.keras.layers.Conv2DTranspose(filters=out_channels, kernel_size=(9,9), strides=1, padding="same")
        self.norm_6 = tf.keras.layers.BatchNormalization()
        self.tanh_out = tf.keras.activations.tanh

        self.resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling="max")

    def build(self, input_shape):
        super(Colorizer, self).build(input_shape)
        self.resnet.build(input_shape=input_shape[1:-1]+(3,),)
        self.resnet.trainable = False

    def call(self, x):
        stack = tf.tile(x, (1,1,1,3))

        x = tf.pad(x, [[0,0], [40,40], [40,40], [0,0]], 'REFLECT')

        x = self.relu_1(self.norm_1(self.conv_1(x)))
        x = self.relu_2(self.norm_2(self.conv_2(x)))
        x = self.relu_3(self.norm_3(self.conv_3(x)))

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)

        #
        # Fuse feature vector onto each 'pixel'
        #

        # extract features from resnet
        feat = self.resnet(stack * 255.0 - 127.5)
        # add (size 1) width, height dimensions
        feat = tf.expand_dims(tf.expand_dims(feat, 1), 1)

        # Stack copy of features across width and height dimensions
        bwh_dim = tf.gather(tf.shape(x), (1,2), axis=0)
        tile_dim = tf.concat([[1,], bwh_dim, [1,]], axis=0)
        
        # duplicate feature vector over width and height of x
        feat = tf.tile(feat, tile_dim)

        # Concatenate feature vectors onto filters
        x = tf.concat([x, feat], axis=-1)
        x = self.fusion_conv(x)

        x = self.relu_4(self.norm_4(self.conv_4(x)))
        x = self.relu_5(self.norm_5(self.conv_5(x)))
        x = self.tanh_out(self.norm_6(self.conv_6(x)))

        return x / 2
