import sys, numpy, time
import mcts
import training_data_io
import tensorflow as tf
from tf_network import Network, find_latest_model

SIZE = mcts.board_size()
KOMI = 7.5

def is_pass(move):
    return move == SIZE * SIZE

def dummy_eval(input_board):
    policy = numpy.random.ranf(SIZE * SIZE + 1).astype(numpy.float32)
    return policy, 0.5

def play_one_game(players):
    moves = []
    passed = False
    current_player = 0
    while True:
        move = players[current_player].gen_play()
        search_count = players[current_player].get_search_count().tolist()
        moves.append(('B' if current_player == 0 else 'W', move, search_count))
        p = is_pass(move)
        if p and passed:
            break
        passed = p
        [p.play(current_player, move) for p in players]
        current_player = 1 - current_player
    print("Score = {}, {}.".format(players[0].score(), players[1].score()), file=sys.stderr)
    return moves, players[0].score()

# def dump_games(filename, games):
#     def translate_move(move):
#         if is_pass(move):
#             return 'pass'
#         char_array = 'abcdefghijklmnopqrstuvwxyz'
#         return '{}{}'.format(char_array[move // SIZE], char_array[move % SIZE])

#     f = open(filename, 'w')
#     f.write('{} {} {}\n'.format(SIZE, KOMI, len(games)))
#     for moves, score in games:
#         f.write('{}\n'.format(len(moves)))
#         for m in moves:
#             f.write('{}:{}'.format(m[0], translate_move(m[1])))
#             for count in m[2]:
#                 f.write(' {}'.format(count))
#             f.write('\n')
#         f.write('{}\n'.format(score))
#     f.close()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

network = Network(find_latest_model())
players = (mcts.MCT(komi=7.5, color=0, eval=network.eval),
           mcts.MCT(komi=7.5, color=1, eval=network.eval))

while True:
    games = []
    for i in range(10):
        games.append(play_one_game(players))
        [p.reset() for p in players]
    training_data_io.store('data/training_data.{}'.format(int(time.time())), games)
