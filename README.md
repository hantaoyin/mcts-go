# mcts-go
An AlphaGo imitation.

Reference:

1. The AlphaGo Zero paper: Mastering the game of Go without human knowledge, Nature volume 550,
pages 354–359 (19 October 2017).

2. leela-zero (https://github.com/leela-zero/leela-zero, an AI Go player using the same
reinforcement learning approach as AlphaGo): Some ideas (but not code) about how to implement the Go
game engine are borrowed from it.

This implementation is intended for educational purposes.  It can only run on a single machine and
will probably be too slow for practical purposes on any Go board size larger than 9x9.  The author
trained a model on a 9x9 Go board using this implementation on a machine with Nvidia GeForce GTX
1080 Ti in about a month.  If you are interested in a more practical implementation that can play
the full size 19x19 Go game, check out leela-zero as mentioned above.
