import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from skimage.segmentation import quickshift
from skimage.filters import threshold_otsu
import cv2
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
import functools
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence

# 0. define top 1, ...

top2_acc = functools.partial(metrics.top_k_categorical_accuracy, k=2)
top3_acc = functools.partial(metrics.top_k_categorical_accuracy, k=3)
top4_acc = functools.partial(metrics.top_k_categorical_accuracy, k=4)
top5_acc = functools.partial(metrics.top_k_categorical_accuracy, k=5)
top2_acc.__name__ = 'top2_acc'
top3_acc.__name__ = 'top3_acc'
top4_acc.__name__ = 'top4_acc'
top5_acc.__name__ = 'top5_acc'

# 1. Define helper functions

def extract_thermal_face(image):
    image_float = image.astype(float) / 255 if image.max() > 1 else image
    segments = quickshift(image_float, ratio=1.0, kernel_size=3, max_dist=6)
    threshold = threshold_otsu(segments)
    binary = segments > threshold
    face_mask = binary.astype(np.uint8) * 255
    face_region = cv2.bitwise_and(image, image, mask=face_mask)
    return face_region

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(72, 96))
    img_array = img_to_array(img)
    thermal_face = extract_thermal_face(img_array)
    thermal_face = np.expand_dims(thermal_face, axis=0) / 255.0
    return thermal_face

# 2. define loss function

centers = tf.Variable(tf.zeros([16, 128]), trainable=False)

def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, 'float32')
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def center_loss(y_true, y_pred):
    alpha = 0.5
    y_true = tf.cast(y_true, 'int32')
    y_true_matrix = tf.one_hot(y_true, depth=16)
    centers_batch = tf.gather(centers, y_true)
    diff = centers_batch - y_pred
    unique_labels, unique_idx, unique_counts = tf.unique_with_counts(y_true)
    appear_times = tf.gather(unique_counts, unique_idx)
    appear_times = tf.reshape(appear_times, (-1, 1))
    diff /= tf.cast((1 + appear_times), tf.float32)
    diff *= alpha
    centers_update = tf.tensor_scatter_nd_sub(centers, tf.reshape(y_true, [-1, 1]), diff)
    with tf.control_dependencies([centers_update]):
        centers_batch_updated = tf.gather(centers, y_true)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - centers_batch_updated), axis=1))
    return loss

def combined_loss(y_true, y_pred):
    # Ensure y_true has the expected shape
    if tf.shape(y_true)[1] != 3:
        tf.print("Error: y_true does not have the expected shape.")
        return 0.0

    # Ensure y_pred has the expected shape (batch_size, 257)
    if tf.shape(y_pred)[1] != 257:
        tf.print("Error: y_pred does not have the expected shape.")
        return 0.0

    # Unpack y_pred
    distance = y_pred[:, 0]
    embedding_1 = y_pred[:, 1:129]
    embedding_2 = y_pred[:, 129:]

    # Calculate losses
    cont_loss = contrastive_loss(y_true[:, 0], distance)
    cent_loss_1 = center_loss(tf.cast(y_true[:, 1], tf.int32), embedding_1)
    cent_loss_2 = center_loss(tf.cast(y_true[:, 2], tf.int32), embedding_2)

    total_loss = cont_loss + 0.1 * (cent_loss_1 + cent_loss_2)
    return total_loss

# 3. create embeddings

base_model = load_model('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/finetuned_thermal_face_vgg16model.h5', custom_objects={
    'top2_acc': top2_acc,
    'top3_acc': top3_acc,
    'top4_acc': top4_acc,
    'top5_acc': top5_acc
})

x = base_model.get_layer('batch_normalization_1').output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
embedding = Dense(128, name='embedding')(x)

embedding_model = Model(inputs=base_model.input, outputs=embedding)

# Print output shapes of all layers
for layer in base_model.layers:
    print(layer.name, layer.output_shape)

# 4. Create the Siamese network for contrastive loss

input_1 = Input(shape=(72, 96, 3))
input_2 = Input(shape=(72, 96, 3))

embedding_1 = embedding_model(input_1)
embedding_2 = embedding_model(input_2)

distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([embedding_1, embedding_2])

siamese_model = Model(inputs=[input_1, input_2], outputs=[distance, embedding_1, embedding_2])

class ContrastiveAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='contrastive_accuracy', **kwargs):
        super(ContrastiveAccuracy, self).__init__(name=name, **kwargs)
        self.correct_counter = self.add_weight(name='correct', initializer='zeros')
        self.total_counter = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assuming y_pred[:, 0] is the distance
        predictions = tf.cast(y_pred[:, 0] < 0.5, tf.float32)
        values = tf.cast(tf.equal(predictions, y_true[:, 0]), tf.float32)
        self.correct_counter.assign_add(tf.reduce_sum(values))
        self.total_counter.assign_add(tf.cast(tf.size(values), tf.float32))

    def result(self):
        return self.correct_counter / self.total_counter

    def reset_state(self):
        self.correct_counter.assign(0)
        self.total_counter.assign(0)

class Rank1Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name='rank1_accuracy', **kwargs):
        super(Rank1Accuracy, self).__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        margin = 1  # or another value depending on how you define rank 1 accuracy
        distances = y_pred[:, 0]
        predictions = tf.cast(distances < margin, dtype=tf.float32)
        correct = tf.cast(tf.equal(predictions, y_true[:, 0]), dtype=tf.float32)
        self.total_correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.cast(tf.size(correct), tf.float32))  # Ensure float32 casting

    def result(self):
        return self.total_correct / self.total

    def reset_state(self):
        self.total_correct.assign(0)
        self.total.assign(0)
class FalseAcceptanceRate(tf.keras.metrics.Metric):
    def __init__(self, name='false_acceptance_rate', **kwargs):
        super(FalseAcceptanceRate, self).__init__(name=name, **kwargs)
        self.false_acceptances = self.add_weight(name='false_acceptances', initializer='zeros')
        self.total_negative = self.add_weight(name='total_negative', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold = 0.5  # Adjust threshold as necessary
        predictions = tf.cast(y_pred[:, 0] < threshold, dtype=tf.float32)
        false_acceptances = tf.logical_and(predictions == 1, y_true[:, 0] == 0)
        self.false_acceptances.assign_add(tf.reduce_sum(tf.cast(false_acceptances, dtype=tf.float32)))
        self.total_negative.assign_add(tf.cast(tf.reduce_sum(tf.cast(y_true[:, 0] == 0, dtype=tf.float32)), tf.float32))

    def result(self):
        return self.false_acceptances / self.total_negative

    def reset_state(self):
        self.false_acceptances.assign(0)
        self.total_negative.assign(0)

class FalseRejectionRate(tf.keras.metrics.Metric):
    def __init__(self, name='false_rejection_rate', **kwargs):
        super(FalseRejectionRate, self).__init__(name=name, **kwargs)
        self.false_rejections = self.add_weight(name='false_rejections', initializer='zeros')
        self.total_positive = self.add_weight(name='total_positive', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold = 0.5  # Adjust threshold
        predictions = tf.cast(y_pred[:, 0] >= threshold, dtype=tf.float32)
        false_rejections = tf.logical_and(predictions == 1, y_true[:, 0] == 1)
        self.false_rejections.assign_add(tf.reduce_sum(tf.cast(false_rejections, dtype=tf.float32)))
        self.total_positive.assign_add(tf.cast(tf.reduce_sum(tf.cast(y_true[:, 0] == 1, dtype=tf.float32)), tf.float32))

    def result(self):
        return self.false_rejections / self.total_positive

    def reset_state(self):
        self.false_rejections.assign(0)
        self.total_positive.assign(0)

# Siamese Model with custom metrics
class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_network, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        self.siamese_network = siamese_network
        self.contrastive_accuracy = ContrastiveAccuracy()
        self.rank1_accuracy = Rank1Accuracy()
        self.false_acceptance_rate = FalseAcceptanceRate()
        self.false_rejection_rate = FalseRejectionRate()

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        self.contrastive_accuracy.update_state(y, y_pred)
        self.rank1_accuracy.update_state(y, y_pred)
        self.false_acceptance_rate.update_state(y, y_pred)
        self.false_rejection_rate.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y, y_pred)

        self.compiled_metrics.update_state(y, y_pred)
        self.contrastive_accuracy.update_state(y, y_pred)
        self.rank1_accuracy.update_state(y, y_pred)
        self.false_acceptance_rate.update_state(y, y_pred)
        self.false_rejection_rate.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super(SiameseModel, self).get_config()
        config.update({
            'siamese_network': tf.keras.utils.serialize_keras_object(self.siamese_network)
        })
        return config

    @classmethod
    def from_config(cls, config):
        siamese_network = tf.keras.utils.deserialize_keras_object(config['siamese_network'])
        return cls(siamese_network=siamese_network)

def create_siamese_network(input_shape):
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    # Load base model with custom accuracy functions
    base_model = load_model('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/finetuned_thermal_face_vgg16model.h5', custom_objects={
        'top2_acc': top2_acc,
        'top3_acc': top3_acc,
        'top4_acc': top4_acc,
        'top5_acc': top5_acc
    })

    x = base_model.get_layer('batch_normalization_1').output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    embedding = Dense(128, name='embedding')(x)

    embedding_model = Model(inputs=base_model.input, outputs=embedding)

    embedding_1 = embedding_model(input_1)
    embedding_2 = embedding_model(input_2)

    distance = Lambda(lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))([embedding_1, embedding_2])
    merged_output = Concatenate()([distance, embedding_1, embedding_2])

    return Model(inputs=[input_1, input_2], outputs=merged_output)

siamese_network = create_siamese_network((72, 96, 3))
siamese_model = SiameseModel(siamese_network)

dummy_input = [tf.zeros((1, 72, 96, 3)), tf.zeros((1, 72, 96, 3))]
_ = siamese_model(dummy_input)

siamese_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=combined_loss,
    metrics=[
        siamese_model.contrastive_accuracy,
        siamese_model.rank1_accuracy,
        siamese_model.false_acceptance_rate,
        siamese_model.false_rejection_rate
    ]
)

siamese_model.summary()

class SiameseGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, dim=(72, 96), n_channels=3):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.image_paths[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 3), dtype=float)

        for i, ID in enumerate(list_IDs_temp):
            X1[i,] = self.load_image(ID)

            # Randomly choose another image
            other_ID = np.random.choice(self.image_paths)
            X2[i,] = self.load_image(other_ID)

            # Set binary label: 1 if same class, 0 if different
            same_class = self.labels[ID]['class_label'] == self.labels[other_ID]['class_label']
            y[i,] = [float(same_class), float(self.labels[ID]['class_label']), float(self.labels[other_ID]['class_label'])]

        return [X1, X2], y

    def load_image(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.dim)
        img = tf.keras.preprocessing.image.img_to_array(img)
        return img / 255.0

# 7. Train the model
def prepare_paths_and_labels(base_dir, n_classes=16):
    paths = []
    labels = {}

    for class_id in range(1, n_classes + 1):
        class_dir = os.path.join(base_dir, f'face{class_id}')
        if not os.path.exists(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                paths.append(img_path)
                labels[img_path] = {
                    "binary_label": 0,  # You need to define how to set the binary label
                    "class_label": class_id - 1  # Assuming class labels are from 0 to n_classes-1
                }
    return paths, labels

# Define base directories for training and validation
train_base_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/train'
val_base_dir = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/val'

train_paths, train_labels = prepare_paths_and_labels(train_base_dir)
val_paths, val_labels = prepare_paths_and_labels(val_base_dir)

train_gen = SiameseGenerator(image_paths=train_paths, labels=train_labels, batch_size=32, dim=(72, 96))
val_gen = SiameseGenerator(image_paths=val_paths, labels=val_labels, batch_size=32, dim=(72, 96))

siamese_model.fit(train_gen, validation_data=val_gen, epochs=10)

results = siamese_model.evaluate(val_gen)
print(f"Validation Loss: {results[0]}, Contrastive Accuracy: {results[1]}")
print(f"Rank-1 Accuracy: {results[2]}, FAR: {results[3]}, FRR: {results[4]}")

siamese_network = siamese_model.siamese_network

# siamese_model.suummary()

base_model = siamese_network.get_layer('model_2')

# Find the embedding layer
embedding_layer = None
for layer in base_model.layers:
    if 'dense' in layer.name and layer.output_shape[-1] == 128:  # Assuming the embedding size is 128
        embedding_layer = layer
        break

siamese_network.summary()

base_model.summary()

embedding_model = Model(inputs=base_model.input, outputs=base_model.get_layer('embedding').output)

# Save this embedding model
embedding_model.save('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/embedding_model.h5')

embedding_model.save('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/embedding_model.keras')

from tensorflow.keras.preprocessing import image
import numpy as np

def prepare_image(file_path, target_size=(72, 96)):
    img = image.load_img(file_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

def prepare_image(file_path, target_size=(72, 96)):
    img = image.load_img(file_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array /= 255.0  # Normalize to [0,1]
    return img_array

model_path = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/embedding_model.keras'
embedding_model.save(model_path)
loaded_embedding_model = tf.keras.models.load_model(model_path)

test_image_path = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/test/face01/0006.jpg'
test_image = prepare_image(test_image_path)

# Predict using the loaded model
embeddings = loaded_embedding_model.predict(test_image)
print("Extracted Embeddings:", embeddings)

# another image to compare against
comparison_image_path = '/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/test/face01/0010.jpg'
comparison_image = prepare_image(comparison_image_path)

# Predict using the loaded model
embeddings = loaded_embedding_model.predict(comparison_image)
print("Extracted Embeddings:", embeddings)

# Get embeddings for both images
embedding_1 = loaded_embedding_model.predict(test_image)
embedding_2 = loaded_embedding_model.predict(comparison_image)

# Calculate Euclidean distance as an example
distance = np.linalg.norm(embedding_1 - embedding_2)
print("Distance between embeddings:", distance)




base_model = siamese_network.get_layer('model_2')

# embedding_model_1 = Model(inputs=base_model.input, outputs=base_model.get_layer('embedding').output)
# embedding_model_1.save('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/embedding_model_1')

# Save the entire Siamese model
siamese_model.save('/content/drive/MyDrive/WORKS/FACIAL RECOGNITION RESEARCH]/data/ExtractedTerravicDatabase_subset/siamese_fx_model')

