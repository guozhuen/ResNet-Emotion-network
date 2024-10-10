import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import glob

def save_image(img, path):
    plt.imshow(img, cmap='jet' if len(img.shape) == 2 else None)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_cam_and_original(model, img, original_img, class_index, filename):
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    final_conv_layer = model.layers[-4]
    model_for_heatmap = tf.keras.models.Model([model.inputs], [final_conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        gtape.watch(img)
        conv_output, predictions = model_for_heatmap(img)
        loss = predictions[:, 0]

    grads = gtape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.squeeze()

    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    #print(heatmap_resized.shape)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    base_name = os.path.basename(filename).split('.')[0]
    save_image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), os.path.join(save_path, f'{base_name}_original.png'))
    save_image(heatmap_resized, os.path.join(save_path, f'{base_name}_cam.png'))
    save_image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),
               os.path.join(save_path, f'{base_name}_superimposed.png'))



model = tf.keras.models.load_model("model_best2.ckpt")
image_folder = "jpg0"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

save_path = "heat3"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for filename in image_files:
    original_img = cv2.imread(filename)
    img = cv2.resize(original_img, (256, 256))
    img = img.astype('float32') / 255.0
    img = img.reshape((1, 256, 256, 3))

    predictions = model.predict(img)
    class_index = int(predictions.squeeze() > 0.5)

    print(f"Generating Class Activation Mapping for image {filename} and class index:", class_index)
    plot_cam_and_original(model, img, original_img, class_index, filename)
