import mcts, uuid, os
import json

__all__ = ['store', 'load']

SIZE = mcts.board_size()
KOMI = 7.5

# Format for games:
# games = [game]
# game = (moves, <score>)
# moves = ['B'|'W', <move>, search_count]
# search_count = [int]

def store(filename, games):
    tmp_fname = '{}.{}'.format(filename, str(uuid.uuid4()))
    f = open(tmp_fname, 'w')
    for moves, score in games:
        f.write('Game: {} {} {} {}\n'.format(SIZE, KOMI, len(moves), score))
        for m in moves:
            f.write('{} {} |'.format(m[0], m[1]))
            assert len(m[2]) == SIZE * SIZE + 1, '{} != {}'.format(m[2], SIZE * SIZE + 1)
            for count in m[2]:
                f.write(' {}'.format(count))
            f.write('\n')
    f.close()
    os.rename(tmp_fname, filename)

def read_one_game(f):
    next_line = f.readline()
    if len(next_line) == 0:
        return None
    if next_line[-1] == '\n':
        next_line = next_line[:-1]
    next_line = next_line.split(' ')
    #0 => 'Game:' (fixed string).
    #1 => SIZE
    #2 => KOMI
    #3 => move count
    #4 => score
    assert len(next_line) == 5, next_line
    assert next_line[0] == 'Game:', next_line
    assert int(next_line[1]) == SIZE, 'Invalid game header, expect SIZE = {}: {}.'.format(SIZE, next_line)
    assert float(next_line[2]) == KOMI, 'Invalid game header, expect KOMI = {}: {}.'.fomrat(KOMI, next_line)
    move_count = int(next_line[3])
    score = float(next_line[4])
    moves = []
    for i in range(move_count):
        move_line = f.readline()
        if move_line[-1] == '\n':
            move_line = move_line[:-1]
        m_str = move_line.split(' ')
        assert len(m_str) == SIZE * SIZE + 4, 'Expecting {} items but got {} from a move line.'.format(SIZE * SIZE + 4, len(m_str))
        moves.append([m_str[0], int(m_str[1]), [int(v) for v in m_str[3:]]])
    return (moves, score)

def load(filename):
    games = []
    with open(filename, 'r') as f:
        while True:
            g = read_one_game(f)
            if g is None:
                return games
            else:
                games.append(g)
