import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# 加载自定义数据集
train_images = np.load('images.npy')
train_labels = np.load('labels.npy')

# 数据预处理
train_images = train_images.astype('float32') / 255.0


num_classes = np.max(train_labels) + 1
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)


num_samples = train_images.shape[0]
train_size = int(num_samples * 0.6)
val_size = int(num_samples * 0.2)

val_images = train_images[train_size:train_size + val_size]
val_labels = train_labels[train_size:train_size + val_size]
test_images = train_images[train_size + val_size:]
test_labels = train_labels[train_size + val_size:]
train_images = train_images[:train_size]
train_labels = train_labels[:train_size]


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(8).shuffle(buffer_size=10000)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(8)

history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)


test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)


def plot_cam_and_original(model, img, original_img, class_index):
    final_conv_layer = model.layers[-4]
    model_for_heatmap = tf.keras.models.Model([model.inputs], [final_conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = model_for_heatmap(img)
        loss = predictions[:, class_index]

    grads = gtape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = heatmap.squeeze()
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("CAM Heatmap")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Superimposed Image")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


img_index = np.random.randint(0, test_images.shape[0])
img = test_images[img_index].reshape((1, 32, 32, 3))
original_img = (test_images[img_index] * 255).astype(np.uint8)
class_index = np.argmax(model.predict(img))

print("Generating Class Activation Mapping for class index:", class_index)

plot_cam_and_original(model, img, original_img, class_index)
