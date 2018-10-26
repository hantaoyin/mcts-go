// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#include <regex>

#include "board.h"
#include "mcts.h"
#include "simple-nn-eval.h"
#include "utils.h"

class InteractivePlayer {
public:
  InteractivePlayer(float komi, go_engine::Color c)
    : board(komi), color(c)
  {}

  go_engine::Move gen_play() {
    CHECK(board.get_next_player() == color);
    LOG(true) << board.DebugString();

    while (true) {
      std::cout << "Move: ";

      std::string move_s;
      std::cin >> move_s;
      if (move_s.empty()) {
        std::cout << std::endl;
        exit(0);
      }
      // Parse the move.
      if (move_s == "pass") {
        return {color};
      }
      std::regex pattern("([a-z])([0-9]+)");
      std::smatch match;
      if (std::regex_match(move_s, match, pattern)) {
        ASSERT(match.size() == 3);
        std::ssub_match x_s = match[1];
        std::ssub_match y_s = match[2];
        ASSERT(x_s.str().size() == 1);
        unsigned col = x_s.str()[0] - 'a';
        unsigned row = std::stoull(y_s.str());
        if (col < go_engine::N && row >= 1 && row <= go_engine::N) {
          go_engine::Move move(color, (row - 1) * go_engine::N + col);
          if (board.is_valid(move)) {
            return move;
          }
        }
      }
      LOG(true) << "Invalid move, try again.";
    }
  }

  void play(go_engine::Move move) {
    board.play(move);
  }

  float score() {
    return color == go_engine::BLACK ? board.score() : -board.score();
  }
private:
  go_engine::BoardInfo board;
  const go_engine::Color color;
};

int main() {
  const std::string network_file = utils::get_latest("data", "network");
  CHECK(!network_file.empty()) << "Failed to find any network file.";
  LOG(true) << "Loading network from " << network_file;
  mcts::Tree<network::SimpleEvalEngine> ai_player(KOMI, go_engine::WHITE, network_file);
  InteractivePlayer interactive_player(KOMI, go_engine::BLACK);

  bool last_move_is_pass = false;
  go_engine::Color current_player = go_engine::BLACK;
  while (true) {
    bool new_move_is_pass;
    if (current_player == go_engine::BLACK) {
      go_engine::Move move = interactive_player.gen_play();
      interactive_player.play(move);
      ai_player.play(move);
      new_move_is_pass = move.pass;
    } else {
      go_engine::Move move = ai_player.gen_play(true);
      ai_player.play(move);
      interactive_player.play(move);
      new_move_is_pass = move.pass;
    }
    current_player = go_engine::opposite_color(current_player);
    if (new_move_is_pass && last_move_is_pass) {
      break;
    }
    last_move_is_pass = new_move_is_pass;
  }
  LOG(true) << std::fixed << std::setprecision(1) << interactive_player.score();

  return 0;
}
