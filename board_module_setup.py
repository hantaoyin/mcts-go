from distutils.core import setup, Extension

cxxargs = ['-std=c++17', '-O3']
ldargs = ['-static-libstdc++']
module1 = Extension('board', sources=['board_py_binding.C', 'Zobrist.C'], depends=['board.h', 'config.h'],
                    extra_compile_args=cxxargs,
                    extra_link_args=ldargs,
                    language='c++')

setup(name='board',
      version='1.0',
      description='Go Board binding 9x9.',
      ext_modules = [module1])
