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

model = Colorizer(scale=16)
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer="adam")
model.build(input_shape=(None,img_nrows,img_ncols,3))
model.summary()

#
# Training on the MS-COCO dataset
#
coco_train, info = tfds.load(name="coco", split="train", with_info=True)

# util functions for data pipeline
def resize(x): return tf.image.resize(x, (img_nrows, img_ncols))
def get_image(x): return x["image"]

#
# Set up tf.data pipeline
#
batch_size = 16
n_samples = info.splits["train"].num_examples
feed = coco_train.repeat()
feed = feed.map(get_image).map(resize)
feed = feed.shuffle(42).batch(batch_size)

# util function to convert a tensor into a valid image
def reshape(x): return x.numpy().reshape((img_nrows, img_ncols, 3)) / 255

i = 0
fig, ax = plt.subplots(3,3)
plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
for batch in feed: 
    color_batch = batch
    # Simple conversion to grayscale by taking a mean of RGB dimension
    gray_batch = tf.reduce_mean(batch, axis=-1, keepdims=True)
    # Copy grayscale data over 3 channels
    gray_batch = tf.tile(gray_batch, (1,1,1,3))

    model.fit(color_batch, gray_batch)
    
    fig.suptitle("Iteration %d" % i)
    if i % 10 == 0:
        for j in range(3):
            plt.subplot(331+3*j)
            plt.cla()
            plt.title("Ground truth")
            plt.imshow(reshape(color_batch[j]))
            plt.axis('off')

            plt.subplot(332+3*j)
            plt.cla()
            plt.title("Grayscale")
            plt.imshow(reshape(gray_batch[j]))
            plt.axis('off')

            plt.subplot(333+3*j)
            plt.cla()
            plt.title("Colorized")
            plt.imshow(reshape(model(tf.expand_dims(gray_batch[j], axis=0))))
            plt.axis('off')
        plt.pause(0.01)

    if i % 1000 == 0:
        model.save_weights("checkpoints/%d" % i)
        plt.savefig("checkpoints/%d.png" % i)
    # train for about 2 epochs
    if i * batch_size > 2 * n_samples:
        break
    i += 1
    