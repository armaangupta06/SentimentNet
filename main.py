import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import os
import pickle  #

# Parameters
max_sequence_length = 500  # Max number of words in a sequence
max_num_words = 20000  # Max number of words to keep in the vocabulary
embedding_dim = 300  # Dimension of the word embeddings
filter_sizes = [3, 4, 5]  # Filter sizes for the convolutional layers
num_filters = 100  # Number of filters per filter size
drop = 0.5  # Dropout rate
batch_size = 50
epochs = 10

# Load the IMDb reviews dataset with text
print("Loading IMDb dataset...")
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

# Extract texts and labels from the dataset
train_texts = []
train_labels = []
for text, label in tfds.as_numpy(train_data):
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)

test_texts = []
test_labels = []
for text, label in tfds.as_numpy(test_data):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(label)

# Tokenize the texts
if os.path.exists('tokenizer.pickle'):
    print("Loading tokenizer...")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    print("Fitting tokenizer...")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(train_texts)
    # Save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

# Convert texts to sequences and pad them
print("Converting texts to sequences and padding...")
X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(np.array(train_labels), num_classes=2)
y_test = to_categorical(np.array(test_labels), num_classes=2)

# Load pre-trained GloVe word embeddings
print("Loading GloVe word embeddings...")
embeddings_index = {}
embedding_path = 'glove.6B.300d.txt'  # Ensure this file is in your working directory

if not os.path.exists(embedding_path):
    print(
        f"{embedding_path} not found. Please download GloVe embeddings from https://nlp.stanford.edu/projects/glove/ and place the file in the working directory.")
else:
    with open(embedding_path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")

    num_words = min(max_num_words, len(word_index) + 1)

    if os.path.exists('embedding_matrix.npy'):
        print("Loading embedding matrix...")
        embedding_matrix = np.load('embedding_matrix.npy')
    else:
        print("Preparing embedding matrix...")
        embedding_matrix = np.zeros((num_words, embedding_dim))

        for word, i in word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        np.save('embedding_matrix.npy', embedding_matrix)
    # Define embedding layers
    embedding_layer_static = Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_sequence_length,
        trainable=False,
    )

    embedding_layer_non_static = Embedding(
        num_words,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_sequence_length,
        trainable=True,
    )


    # Function to build the CNN model
    def build_model(embedding_layer, multichannel=False):
        sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
        if multichannel:
            # For multichannel, embedding_layer is a list
            embedded_sequences = [layer(sequence_input) for layer in embedding_layer]
            embedded_sequences = Concatenate()(embedded_sequences)
        else:
            embedded_sequences = embedding_layer(sequence_input)
        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(
                filters=num_filters,
                kernel_size=sz,
                activation='relu'
            )(embedded_sequences)
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = Dropout(drop)(z)
        z = Dense(2, activation='softmax')(z)
        model = Model(sequence_input, z)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model


    # Build, train, and save or load CNN-static model
    if os.path.exists('model_static.h5'):
        print("Loading pre-trained CNN-static model...")
        model_static = load_model('model_static.h5')
    else:
        print("Building CNN-static model...")
        model_static = build_model(embedding_layer_static)
        print(model_static.summary())
        print("Training CNN-static model...")
        model_static.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=2
        )
        model_static.save('model_static.h5')

    # Build, train, and save or load CNN-non-static model
    if os.path.exists('model_non_static.h5'):
        print("Loading pre-trained CNN-non-static model...")
        model_non_static = load_model('model_non_static.h5')
    else:
        print("Building CNN-non-static model...")
        model_non_static = build_model(embedding_layer_non_static)
        print(model_non_static.summary())
        print("Training CNN-non-static model...")
        model_non_static.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=2
        )
        model_non_static.save('model_non_static.h5')

    # Build, train, and save or load CNN-multichannel model
    if os.path.exists('model_multichannel.h5'):
        print("Loading pre-trained CNN-multichannel model...")
        model_multichannel = load_model('model_multichannel.h5')
    else:
        print("Building CNN-multichannel model...")
        model_multichannel = build_model(
            [embedding_layer_static, embedding_layer_non_static],
            multichannel=True
        )
        print(model_multichannel.summary())
        print("Training CNN-multichannel model...")
        model_multichannel.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=2
        )
        model_multichannel.save('model_multichannel.h5')

    # Evaluate the models
    print("Evaluating models on test data...")
    loss_static, acc_static = model_static.evaluate(X_test, y_test, verbose=0)
    loss_non_static, acc_non_static = model_non_static.evaluate(X_test, y_test, verbose=0)
    loss_multichannel, acc_multichannel = model_multichannel.evaluate(X_test, y_test, verbose=0)

    print(f"CNN-static model accuracy: {acc_static:.4f}")
    print(f"CNN-non-static model accuracy: {acc_non_static:.4f}")
    print(f"CNN-multichannel model accuracy: {acc_multichannel:.4f}")

    # Example of using the saved models for inference
    new_texts = ["This movie was fantastic!", "I did not enjoy the film.", "The rain was falling fast and the drops sunk into the ground.", "The girl saw Roshan and walked away."]
    new_sequences = tokenizer.texts_to_sequences(new_texts)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_length)

    # Predict with the static model
    predictions = model_static.predict(new_padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)

    # Display results
    for text, label in zip(new_texts, predicted_classes):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

