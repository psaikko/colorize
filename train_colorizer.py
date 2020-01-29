import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from colorizer import Colorizer

# Workaround for tf 2.0 issue
# https://stackoverflow.com/a/58684421
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_nrows = 256
img_ncols = 256

model = Colorizer(scale=16, out_channels=2)
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer="adam")
model.build(input_shape=(None,img_nrows,img_ncols,1))
model.summary()

#
# Training on the MS-COCO dataset
#
coco_train, info = tfds.load(name="coco", split="train", with_info=True)

print(info)

# util functions for data pipeline
def resize(x): return tf.image.resize(x, (img_nrows, img_ncols))
def get_image(x): return x["image"]
def make_class_filter(class_id):
    def class_filter(x):
        return tf.reduce_any(x["objects"]["label"] == class_id)
    return class_filter

#
# Set up tf.data pipeline
#
batch_size = 16
STOPSIGN_ID = 11
n_samples = info.splits["train"].num_examples
feed = coco_train.filter(make_class_filter(STOPSIGN_ID))
feed = feed.map(get_image, num_parallel_calls=4).map(resize)
feed = feed.shuffle(43).repeat().batch(batch_size)
# util function to convert a tensor into a valid image
def reshape(x,channels=3): return x.numpy().reshape((img_nrows, img_ncols, channels)) / 255

i = 0
fig, ax = plt.subplots(4,3)
plt.subplots_adjust(left=0,right=1,bottom=0,top=0.9,wspace=0,hspace=0.2)
for batch in feed: 

    yuv_batch = tf.image.rgb_to_yuv(batch / 255.0)

    y_batch = tf.expand_dims(yuv_batch[:,:,:,0], axis=-1)
    uv_batch = yuv_batch[:,:,:,1:]

    model.fit(y_batch, uv_batch)
    
    if i % 10 == 0:
        j = 0

        for k in range(12):
            plt.subplot(4,3,k+1)
            plt.cla()
            plt.axis('off')

        fig.suptitle("Iteration %d" % i)
        plt.subplot(430+2)
        plt.title("Image")
        plt.imshow(reshape(batch[j]))

        plt.subplot(430+3+1)
        plt.title("Y")
        plt.imshow(tf.squeeze(y_batch[j]).numpy(), cmap='Greys', vmin=0, vmax=1)

        plt.subplot(430+3+2)
        plt.title("U")
        plt.imshow(tf.squeeze(uv_batch[j,:,:,0]).numpy(), vmin=0, vmax=1)

        plt.subplot(430+3+3)
        plt.title("V")
        plt.imshow(tf.squeeze(uv_batch[j,:,:,1]).numpy(), vmin=0, vmax=1)

        uv_pred = model(tf.expand_dims(y_batch[j], axis=0))

        plt.subplot(430+3*2+2)
        plt.title("U'")
        plt.imshow(tf.squeeze(uv_pred[0,:,:,0]).numpy(), vmin=0, vmax=1)

        plt.subplot(430+3*2+3)
        plt.title("V'")
        plt.imshow(tf.squeeze(uv_pred[0,:,:,1]).numpy(), vmin=0, vmax=1)

        plt.subplot(4,3,11)
        plt.title("Colorized")
        yuv_pred = tf.concat([y_batch[0],uv_pred[0]], axis=-1)
        rgb_pred = tf.image.yuv_to_rgb(yuv_pred)
        plt.imshow(reshape(rgb_pred,3)*255.0)

        plt.pause(0.01)

    if i % 1000 == 0:
        model.save_weights("checkpoints/%d" % i)
        plt.savefig("checkpoints/%d.png" % i)
    # train for about 2 epochs
    if i * batch_size > 2 * n_samples:
        break
    i += 1
    