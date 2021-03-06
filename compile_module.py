from distutils.core import setup, Extension

cxxargs = ['-std=c++17', '-O3']
ldargs = []

modules = [
    Extension('mcts',
              sources=['mcts_py_binding.C', 'Zobrist.C'],
              depends=['board.h', 'config.h', 'debug_msg.h', 'mcts.h', 'eval_bridge.h'],
              extra_compile_args=cxxargs, extra_link_args=ldargs, language='c++'),
    Extension('board',
              sources=['board_py_binding.C', 'Zobrist.C'],
              depends=['board.h', 'config.h', 'debug_msg.h'],
              extra_compile_args=cxxargs, extra_link_args=ldargs, language='c++'),
]

setup(name='mcts',
      version='1.0',
      description='Go board engine and MCTS modules.',
      ext_modules = modules)
