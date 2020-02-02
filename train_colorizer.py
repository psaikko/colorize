import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorizer import Colorizer

# Workaround for tf 2.0 issue
# https://stackoverflow.com/a/58684421
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_nrows = 256
img_ncols = 256

model = Colorizer(scale=16, out_channels=2)
opt = tf.optimizers.Adam()
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=opt)
model.build(input_shape=(None,img_nrows,img_ncols,1))
model.summary()

#
# Training on the MS-COCO dataset
#
coco_train, info = tfds.load(name="coco", split="train", with_info=True)

# util functions for data pipeline
def resize(x): return tf.image.resize(x, (img_nrows, img_ncols))
def get_image(x): return x["image"]
def make_class_filter(class_id):
    def class_filter(x):
        return tf.reduce_any(x["objects"]["label"] == class_id)
    return class_filter
def to_yuv(x): return tf.image.rgb_to_yuv(x / 255.0)

#
# Set up tf.data pipeline
#
data_batch_size = 12
STOPSIGN_ID = 11
n_samples = info.splits["train"].num_examples
feed = coco_train 
feed = feed.map(get_image, num_parallel_calls=4)\
           .map(resize, num_parallel_calls=4)\
           .map(to_yuv, num_parallel_calls=4)\
           .batch(data_batch_size)

subplots_w = 4
subplots_h = 3
n_subplots = subplots_w * subplots_h
fig, ax = plt.subplots(subplots_h, subplots_w)
plt.subplots_adjust(left=0,right=1,bottom=0,top=0.9,wspace=0,hspace=0.2)

example_index = 0

def show_rgb(x):
    plt.imshow(x.numpy().reshape((img_nrows, img_ncols, 3)), vmin=0, vmax=255)

def show_yuv(x):
    x = tf.image.yuv_to_rgb(x)
    x = tf.clip_by_value(x,0,1)
    plt.imshow(x.numpy().reshape((img_nrows, img_ncols, 3)), vmin=0, vmax=1)

def show_2ch(x):
    plt.imshow(tf.squeeze(x).numpy(), vmin=-0.5, vmax=0.5)

def show_1ch(x):
    plt.imshow(tf.squeeze(x).numpy(), cmap='Greys', vmin=0, vmax=1)

i = 0
epochs = 2
for yuv_images in feed:
    y_data = tf.expand_dims(yuv_images[:,:,:,0], axis=-1)
    uv_data = yuv_images[:,:,:,1:]

    model.fit(y_data, uv_data, batch_size=8)
    uv_pred = model(y_data[:n_subplots])
    yuv_pred = tf.concat([y_data[:n_subplots],uv_pred], axis=-1)

    plt.suptitle("Iteration %d" % i)

    for k in range(n_subplots):
        plt.subplot(subplots_h,subplots_w,k+1)
        plt.cla()
        plt.axis('off')
        show_yuv(yuv_pred[k,:,:,:])

    plt.pause(0.01)

    if i % 10 == 0:
        model.save_weights("checkpoints/fusion%d" % i)
        plt.savefig("checkpoints/fusion%d.png" % i)
    i += 1

    if (i * data_batch_size) >= epochs * n_samples:
        break
    