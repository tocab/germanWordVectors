import tensorflow as tf
import pandas as pd
import numpy as np
import sys, spacy, argparse
from model import model

MAX_SEQUENCE_LENGTH = 50
COUNT_VAL_DATA = 500
BATCH_SIZE = 64

pd.set_option('display.expand_frame_repr', False)

# Load spacy model
nlp = spacy.load("de")

# Session config
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def main(args):
    data = "data/downloaded.tsv"
    data_pd = pd.read_csv(data, sep="\t", header=None, names=["id", "sentiment", "hash", "rt", "text"])
    data_pd = data_pd[["text", "sentiment"]]
    data_pd = data_pd[data_pd["text"] != "Not Available"]
    data_pd = data_pd.reset_index(drop=True)

    classes_dict = {"negative": 0,
                    "neutral": 1,
                    "positive": 2}

    data_pd["sentiment"] = data_pd["sentiment"].map(classes_dict)

    if args.model == "spacy_context":
        spacy_context_eval(data_pd)
    elif args.model == "elmo":
        elmo_eval(data_pd)
    elif args.model == "fasttext":
        fasttext_eval(data_pd)
    elif args.model == "fasttext_quantized":
        fasttext_quantized_eval(data_pd)
    else:
        print("model not found")
        sys.exit()


def fasttext_eval(data_pd):
    model_path = "vectors/fasttext/cc.de.300.bin"
    tokenized_sentences = tokenize_sentences(data_pd["text"])

    x_data = fasttext_features(tokenized_sentences, model_path)
    y_data = data_pd["sentiment"].values

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data)

    with tf.Session(config=config) as sess:
        # create model
        a_model = model.prediction_model(max_seq_len=x_train.shape[1], embedding_size=x_train.shape[2],
                                         batch_size=BATCH_SIZE,
                                         sess=sess)
        a_model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        a_model.eval_model(x_val, y_val)


def fasttext_quantized_eval(data_pd):
    model_path = "vectors/fasttext/cc.de.300_quantized.ftz"
    tokenized_sentences = tokenize_sentences(data_pd["text"])

    x_data = fasttext_features(tokenized_sentences, model_path)
    y_data = data_pd["sentiment"].values

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data)

    with tf.Session(config=config) as sess:
        # create model
        a_model = model.prediction_model(max_seq_len=x_train.shape[1], embedding_size=x_train.shape[2],
                                         batch_size=BATCH_SIZE,
                                         sess=sess)
        a_model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        a_model.eval_model(x_val, y_val)


def fasttext_features(sentences, model):
    import fastText
    ft_model = fastText.load_model(model)

    x_data = []
    for sentence in sentences:
        word_vectors = [ft_model.get_sentence_vector(word) for word in sentence]
        x_data.append(word_vectors)

    x_data = pad_data(x_data)

    return np.array(x_data)


def spacy_context_eval(data_pd):
    tokenized_sentences = tokenize_sentences(data_pd["text"], return_vectors=True)
    x_data = pad_data(tokenized_sentences)
    y_data = data_pd["sentiment"].values

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data)

    with tf.Session(config=config) as sess:
        # create model
        a_model = model.prediction_model(max_seq_len=x_train.shape[1], embedding_size=x_train.shape[2],
                                         batch_size=BATCH_SIZE,
                                         sess=sess)
        a_model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        a_model.eval_model(x_val, y_val)


def elmo_eval(data_pd):
    from bilm import Batcher, BidirectionalLanguageModel, weight_layers
    # Location of pretrained LM.  Here we use the test fixtures.
    vocab_file = "vectors/elmo/hdf5/vocab.txt"
    options_file = "vectors/elmo/hdf5/options.json"
    weight_file = "vectors/elmo/hdf5/weights.hdf5"

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, 50)

    # Input placeholders to the biLM.
    character_ids = tf.placeholder('int32', shape=(None, None, 50))

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file)

    # Get ops to compute the LM embeddings.
    embeddings = bilm(character_ids)

    elmo_input = weight_layers('input', embeddings, l2_coef=0.0)

    tokenized_sentences = tokenize_sentences(data_pd["text"])

    with tf.Session(config=config) as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        # Create batches of data.
        ids = batcher.batch_sentences(tokenized_sentences)

        batch_size = 64
        beginning_index = 0
        x_data = []
        while beginning_index <= ids.shape[0]:
            # Compute ELMo representations (here for the input only, for simplicity).
            elmo_vectors = sess.run(elmo_input['weighted_op'],
                                    feed_dict={character_ids: ids[beginning_index:beginning_index + batch_size, :, :]})
            x_data.append(elmo_vectors)
            beginning_index += batch_size

        x_data = np.concatenate(x_data)

    y_data = data_pd["sentiment"].values

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data)

    with tf.Session(config=config) as sess:
        # create model
        a_model = model.prediction_model(max_seq_len=x_train.shape[1], embedding_size=x_train.shape[2],
                                         batch_size=BATCH_SIZE,
                                         sess=sess)
        a_model.train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
        a_model.eval_model(x_val, y_val)


def train_test_split(x_data, y_data):
    x_train = x_data[:-COUNT_VAL_DATA, :, :]
    x_val = x_data[x_train.shape[0]:, :, :]
    y_train = y_data[:-COUNT_VAL_DATA]
    y_val = y_data[y_train.shape[0]:]

    return x_train, x_val, y_train, y_val


def pad_data(data):
    x_data = []
    for words in data:
        words = np.array(words)
        words = np.pad(words, [[0, MAX_SEQUENCE_LENGTH], [0, 0]], mode="constant")
        words = words[:MAX_SEQUENCE_LENGTH, :]
        x_data.append(words)

    x_data = np.array(x_data)
    return x_data


def tokenize_sentences(sentences, return_vectors=False):
    tokenized_sentences = []
    for doc in nlp.pipe(sentences, batch_size=256):
        words = []
        for word in doc:
            if word.has_vector:
                if return_vectors:
                    words.append(word.vector)
                else:
                    words.append(word.text)
        tokenized_sentences.append(words)
    return tokenized_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model that will be evaluated')
    args = parser.parse_args()
    main(args)
