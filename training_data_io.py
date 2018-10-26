import mcts, uuid, os
import json

__all__ = ['store', 'load']

SIZE = mcts.board_size()
KOMI = 7.5

def store(filename, games):
    training_data = {
        'size': SIZE,
        'komi': KOMI,
        'games': games,
    }
    tmp_fname = '{}.{}'.format(filename, str(uuid.uuid4()))
    f = open(tmp_fname, 'w')
    json.dump(training_data, f)
    f.close()
    os.rename(tmp_fname, filename)

def load(filename):
    f = open(filename, 'r')
    return json.load(f)
