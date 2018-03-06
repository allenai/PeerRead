
class CNNConfig(object):
    embedding_dim = 300
    seq_length = 200 #change by input
    num_classes = 9 #change by input
    vocab_size = 20000 #change by input
    num_filters = 128
    kernel_size = 5
    hidden_dim = 100
    dropout_keep_prob = 1.0
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100
    print_per_batch = 50
    save_per_batch = 10


class RNNConfig(object):
    embedding_dim = 300
    seq_length = 200 #change by input
    num_classes = 9 #change by input
    vocab_size = 20000 #change by input
    num_layers= 1
    hidden_dim = 100
    rnn = 'gru'
    dropout_keep_prob = 1.0
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100
    print_per_batch = 50
    save_per_batch = 10

class DANConfig(object):
    embedding_dim = 300
    seq_length = 200 #change by input
    num_classes = 9 #change by input
    vocab_size = 20000 #change by input
    num_layers= 1
    hidden_dim = 100
    dropout_keep_prob = 1.0
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100
    print_per_batch = 50
    save_per_batch = 10


