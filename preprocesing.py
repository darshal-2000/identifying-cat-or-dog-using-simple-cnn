import tensorflow as tf
def dataset_preprocess(path):
    dataset = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=32)
    normalized_dataset = dataset.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label))
    final_datset = normalized_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return final_datset

def single_image_preprocess(path):
    img = tf.keras.utils.load_img(path, target_size=(224,224))
    img_tensor = tf.keras.utils.img_to_array(img)
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.0
    return img_tensor