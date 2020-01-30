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
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer="adam")
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
batch_size = 16
STOPSIGN_ID = 11
n_samples = info.splits["train"].num_examples
feed = coco_train.filter(make_class_filter(STOPSIGN_ID))
feed = feed.map(get_image, num_parallel_calls=4)\
           .map(resize, num_parallel_calls=4)\
           .map(to_yuv, num_parallel_calls=4)

yuv_images = []

#
# Process data prior to training
#
for x in tqdm(feed):
    x = tf.expand_dims(x, axis=0)
    yuv_images += [x]
yuv_images = tf.concat(yuv_images, axis=0)
print(yuv_images.shape)
y_data = tf.expand_dims(yuv_images[:,:,:,0], axis=-1)
uv_data = yuv_images[:,:,:,1:]

# util function to convert a tensor into a valid image
def reshape(x,channels=3): return x.numpy().reshape((img_nrows, img_ncols, channels)) / 255

epochs = 100
fig, ax = plt.subplots(4,3)
plt.subplots_adjust(left=0,right=1,bottom=0,top=0.9,wspace=0,hspace=0.2)

example_index = 0

for i in range(epochs):
    model.fit(y_data, uv_data, batch_size=16)

    for k in range(12):
        plt.subplot(4,3,k+1)
        plt.cla()
        plt.axis('off')

    fig.suptitle("Epoch %d" % (i+1))
    plt.subplot(430+2)
    plt.title("Image")
    plt.imshow(reshape(tf.image.yuv_to_rgb(yuv_images[example_index]) * 255))

    plt.subplot(430+3+1)
    plt.title("Y")
    plt.imshow(tf.squeeze(y_data[example_index]).numpy(), cmap='Greys', vmin=0, vmax=1)

    plt.subplot(430+3+2)
    plt.title("U")
    plt.imshow(tf.squeeze(uv_data[example_index,:,:,0]).numpy(), vmin=-0.5, vmax=0.5)

    plt.subplot(430+3+3)
    plt.title("V")
    plt.imshow(tf.squeeze(uv_data[example_index,:,:,1]).numpy(), vmin=-0.5, vmax=0.5)

    uv_pred = model(tf.expand_dims(y_data[example_index], axis=0))

    plt.subplot(430+3*2+2)
    plt.title("U'")
    plt.imshow(tf.squeeze(uv_pred[0,:,:,0]).numpy(), vmin=-0.5, vmax=0.5)

    plt.subplot(430+3*2+3)
    plt.title("V'")
    plt.imshow(tf.squeeze(uv_pred[0,:,:,1]).numpy(), vmin=-0.5, vmax=0.5)

    plt.subplot(4,3,11)
    plt.title("Colorized")
    yuv_pred = tf.concat([y_data[0],uv_pred[0]], axis=-1)
    rgb_pred = tf.image.yuv_to_rgb(yuv_pred)
    plt.imshow(reshape(rgb_pred,3)*255)

    plt.pause(0.01)

    if i % 10 == 0:
        model.save_weights("checkpoints/%d" % i)
        plt.savefig("checkpoints/%d.png" % i)
    