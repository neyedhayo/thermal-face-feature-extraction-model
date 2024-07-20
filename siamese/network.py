class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_network, **kwargs):
        super(SiameseModel, self).__init__(**kwargs)
        self.siamese_network = siamese_network

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

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        # Save the siamese_network configuration instead of the entire object
        config = super(SiameseModel, self).get_config()
        config.update({
            'siamese_network': tf.keras.utils.serialize_keras_object(self.siamese_network)
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the siamese_network configuration to a Keras model
        siamese_network = tf.keras.utils.deserialize_keras_object(config['siamese_network'])
        return cls(siamese_network=siamese_network)

# Create the Siamese network
def create_siamese_network(input_shape):
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    base_model = load_model('/content/drive/MyDrive/finetuned_thermal_face_vgg16model.h5', custom_objects={
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

    distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([embedding_1, embedding_2])

    merged_output = Concatenate()([distance, embedding_1, embedding_2])

    return Model(inputs=[input_1, input_2], outputs=merged_output)