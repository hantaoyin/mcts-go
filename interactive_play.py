import sys
import threading

import board
import mcts
import tensorflow as tf
from tf_network import Network

SIZE = mcts.board_size()

def is_pass(move):
    return move == SIZE * SIZE

class InteractivePlayer:
    def __init__(self, *,
                 komi,
                 color):
        self.board = board.Board(komi)
        self.my_color = color

    def _decode_move(self, move):
        move.rstrip('\n')
        if move == 'pass':
            return SIZE * SIZE
        err = ValueError(f'Invalid move {move}, must be pass or a lower case letter (column) followed by a number (row).')
        if len(move) != 2:
            raise err
        row = int(move[1]) - 1
        col = ord(move[0]) - ord('a')
        if row >= 0 and row < SIZE and col >= 0 and col < SIZE:
            mid = row * SIZE + col
            if self.board.is_valid(self.my_color, mid):
                return row * SIZE + col
        raise err

    def gen_play(self, _):
        while True:
            print(self.board.debug())
            print(f'Enter your move: ', end='')
            move = input()
            try:
                mid = self._decode_move(move)
                break
            except ValueError as e:
                print(e)
                pass
        return mid

    def play(self, color, move):
        self.board.play(color, move)

    def reset(self):
        sys.exit()


def play_one_game(players, debug_log):
    passed = False
    current_player = 0
    while True:
        move = players[current_player].gen_play(debug_log)
        p = is_pass(move)
        if p and passed:
            break
        passed = p
        [p.play(current_player, move) for p in players]
        current_player = 1 - current_player
    if debug_log:
        print("Score = {}, {}.".format(players[0].score(), players[1].score()))


class WorkerThread(threading.Thread):
    def __init__(self, thread_id, eval_object):
        super().__init__()
        self.daemon = True
        self.thread_id = thread_id
        self.eval_object = eval_object

    def run(self):
        players = None
        if self.thread_id == 0:
            while True:
                print(f'Play black (B) or white (W)?')
                color = input()
                if color in ('B', 'W'):
                    break
                else:
                    print('Invalid color.')
            if color == 'B':
                players = (InteractivePlayer(komi=7.5, color=0),
                           mcts.Tree(komi=7.5, color=1, eval=self.eval_object))
            else:
                players = (mcts.Tree(komi=7.5, color=0, eval=self.eval_object),
                           InteractivePlayer(komi=7.5, color=1))
        else:
            # All other threads are fillers since eval bridge requires multiple threads to fill each
            # batch.
            players = (mcts.Tree(komi=7.5, color=0, eval=self.eval_object),
                       mcts.Tree(komi=7.5, color=1, eval=self.eval_object))
        while True:
            play_one_game(players, self.thread_id == 0)
            if self.thread_id == 0:
                if isinstance(players[0], mcts.Tree):
                    score = players[0].score()
                else:
                    score = -players[1].score()
                print(f'Score: B = {score}, W = {-score}.')
            [p.reset() for p in players]

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
