# -*- coding:utf-8-unix -*-
# ==================================================================================================
test-all: board-5x5

board-5x5: ../board.h ../config.h ../debug_msg.h board-5x5.C ../Zobrist.C
	g++ -std=c++17 -O3 -Wall -Wextra board-5x5.C ../Zobrist.C -I.. -o board-5x5
	./board-5x5 && echo "All pass."

clean:
	-rm board-5x5
