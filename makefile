all: self_play init_network train interactive_play

self_play: config.h debug_msg.h board.h utils.h training.h simple-nn-eval.h mcts.h main_self_play.C Zobrist.C
	g++ -std=c++17 -O3 -Wall -Wextra main_self_play.C Zobrist.C -lstdc++fs -o self_play

init_network: config.h debug_msg.h board.h training.h utils.h simple-nn-eval.h main_init_network.C Zobrist.C
	g++ -std=c++17 -O3 -Wall -Wextra main_init_network.C Zobrist.C -o init_network

train: config.h debug_msg.h board.h training.h utils.h simple-nn-eval.h mcts.h main_train.C Zobrist.C
	g++ -std=c++17 -O3 -Wall -Wextra main_train.C Zobrist.C -lstdc++fs -o train

interactive_play: config.h debug_msg.h board.h training.h utils.h simple-nn-eval.h main_interactive_play.C Zobrist.C
	g++ -std=c++17 -O3 -Wall -Wextra main_train.C Zobrist.C -lstdc++fs -o interactive_play

clean:
	-rm self_play init_network train interactive_play
	-rm *.exe *.exe.stackdump
