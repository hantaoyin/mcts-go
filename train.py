import numpy, time, glob
import mcts, board
import tensorflow as tf
from tf_network import Network
import training_data_io

SIZE = mcts.board_size()
KOMI = 7.5

def transform_training_data(games):
    x = []
    y0 = []
    y1 = []
    for g in games:
        moves = g[0]
        score = g[1]
        b = board.Board(KOMI)
        for m in moves:
            color = 0 if m[0] == 'B' else 1
            move = m[1]
            network_input = numpy.empty([3, SIZE, SIZE], dtype=numpy.float32)
            for pos_x in range(SIZE):
                for pos_y in range(SIZE):
                    network_input[0, pos_x, pos_y] = b.has_stone(color, pos_x * SIZE + pos_y)
                    network_input[1, pos_x, pos_y] = b.has_stone(1 - color, pos_x * SIZE + pos_y)
                    network_input[2, pos_x, pos_y] = color

            assert(len(m[2]) == SIZE * SIZE + 1)
            network_output_policy = numpy.array(m[2], dtype=numpy.float32)
            network_output_policy += 1.e-5
            network_output_policy = network_output_policy / numpy.sum(network_output_policy)
            network_output_value = 1.e-5 + numpy.array([color if score < 0 else 1 - color], dtype=numpy.float32) * (1. - 1.e-5 * 2.)
            b.play(color, move)
            x.append(network_input)
            y0.append(network_output_policy)
            y1.append(network_output_value)
        if score != b.score():
            raise ValueError('Score mismatch: {} (file) !=  {} (board).'.format(b.score(), score))
    return numpy.stack(x), [numpy.stack(y0), numpy.stack(y1)]

def load_training_data():
    training_data = sorted(glob.glob('data/training_data.*'))
    all_games = sum([training_data_io.load(v) for v in training_data], [])
    # Transform into a format useable for model training.
    return transform_training_data(all_games)

x, y = load_training_data()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)):
    network = Network()
    print(network.model.summary())
    network.fit(x, y, epochs=2)
    network.store('data/network.{}'.format(int(time.time())))
