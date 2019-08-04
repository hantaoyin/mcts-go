import sys
import numpy
import time
import threading

import mcts, training_data_io
import tensorflow as tf
from tf_network import Network

SIZE = mcts.board_size()

def is_pass(move):
    return move == SIZE * SIZE

def dummy_eval(input_board):
    policy = numpy.random.ranf(SIZE * SIZE + 1).astype(numpy.float32)
    return policy, 0.5

def play_one_game(players, debug_log):
    moves = []
    passed = False
    current_player = 0
    while True:
        move = players[current_player].gen_play(debug_log)
        search_count = players[current_player].get_search_count().tolist()
        moves.append(('B' if current_player == 0 else 'W', move, search_count))
        p = is_pass(move)
        if p and passed:
            break
        passed = p
        [p.play(current_player, move) for p in players]
        current_player = 1 - current_player
    if debug_log:
        print("Score = {}, {}.".format(players[0].score(), players[1].score()))
    return moves, players[0].score()

class WorkerThread(threading.Thread):
    def __init__(self, thread_id, eval_object):
        super().__init__()
        self.daemon = True
        self.thread_id = thread_id
        self.eval_object = eval_object

    def run(self):
        players = (mcts.Tree(komi=7.5, color=0, eval=self.eval_object),
                   mcts.Tree(komi=7.5, color=1, eval=self.eval_object))
        while True:
            games = []
            for i in range(10):
                games.append(play_one_game(players, self.thread_id == 0))
                [p.reset() for p in players]
            training_data_io.store('data/training_data.{}.{}'.format(self.thread_id, int(time.time())), games)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

network = Network()
eval_object = mcts.EvalBridge(network.eval)

worker_threads = [WorkerThread(i, eval_object) for i in range(eval_object.worker_thread_count())]
for w in worker_threads:
    w.start()

# The following is an infinite loop.
#
# This must be in the same thread that builds the Tensorflow model (which is the main thread),
# otherwise I get runtime crashes.
eval_object.start_eval()
