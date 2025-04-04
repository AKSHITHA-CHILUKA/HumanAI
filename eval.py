import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define CNN feature extractor
def build_cnn(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(1,2))(x)  # Keep height, reduce width
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(1,2))(x)
    x = layers.Reshape(target_shape=(-1, 512))(x)  # Flatten height for LSTM
    return keras.Model(inputs, x, name='CNN_FeatureExtractor')

# Define BiLSTM + CTC model
def build_ocr_model(input_shape, vocab_size):
    inputs = keras.Input(shape=input_shape)
    cnn_output = build_cnn(input_shape)(inputs)
    
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(cnn_output)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dense(vocab_size + 1, activation='softmax')(x)  # +1 for CTC blank token
    
    return keras.Model(inputs, x, name='OCR_Model')

# Define CTC loss function
def ctc_loss_lambda(y_true, y_pred):
    input_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(y_pred, 0), tf.int32), axis=-1)
    label_length = tf.math.reduce_sum(tf.cast(tf.math.not_equal(y_true, 0), tf.int32), axis=-1)
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Example usage
input_shape = (32, 128, 1)  # Height, Width, Channels (grayscale)
vocab_size = 50  # Example vocabulary size (adjust based on dataset)
ocr_model = build_ocr_model(input_shape, vocab_size)
ocr_model.compile(optimizer='adam', loss=ctc_loss_lambda)
ocr_model.summary()

