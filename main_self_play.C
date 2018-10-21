// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
// Each saved file contains the latest 100000 games.  This means 2 training data files can have
// overlapping examples.  This is done so that we can read all needed training data from a single
// file.
#include <algorithm>
#include <array>
#include <bitset>
#include <deque>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <mmintrin.h>

#include "config.h"
#include "debug_msg.h"
#include "board.h"
#include "training.h"
#include "simple-nn-eval.h"
#include "mcts.h"
#include "utils.h"

class GenTrainingData {
public:
  static constexpr size_t maxGames = 10000000;

  GenTrainingData(const std::string& _working_dir): working_dir(_working_dir) {
  }

  std::string merge_existing() {
    // Read existing training data.
    const std::string &latest_training = utils::get_latest(working_dir, "training");
    if (!latest_training.empty()) {
      LOG(true) << "Mering training data from " << latest_training;
      std::ifstream input(latest_training);
      CHECK(input.is_open()) << "Failed to open " << latest_training << ".";

      size_t board_size;
      float komi;
      size_t n_games;
      input >> board_size >> komi >> n_games;
      CHECK(board_size == BOARD_SIZE) << board_size << " != " << BOARD_SIZE;
      CHECK((komi - KOMI) < 1.e-7f * KOMI) << komi << " != " << KOMI;
      CHECK(n_games < 10000000ULL) << n_games;
      for (size_t i = 0; i < n_games; ++i) {
        data.emplace_back(input);
      }
    }
    return latest_training;
  }

  void play(size_t count) {
    const std::string network_file = utils::get_latest(working_dir, "network");
    CHECK(!network_file.empty()) << "Failed to find any network file.";
    LOG(true) << "Loading network from " << network_file;
    std::array<mcts::Tree<network::SimpleEvalEngine>, 2> players{{
        {KOMI, go_engine::BLACK, network_file},
        {KOMI, go_engine::WHITE, network_file}
      }};

    for (size_t i = 0; i < count; ++i) {
      players[0].reset();
      players[1].reset();

      bool debug_log = i % 10 == 0;
      data.emplace_back();
      training::Game& new_game = data.back();

      bool last_move_is_pass = false;
      go_engine::Color current_player = go_engine::BLACK;
      while (true) {
        go_engine::Move move = players[current_player].gen_play(debug_log);
        const auto& count(players[current_player].get_search_count());
        players[current_player].play(move);
        go_engine::Color opponent = go_engine::opposite_color(current_player);
        players[opponent].opponent_play(move);
        current_player = opponent;
        new_game.states.emplace_back(move, count);
        if (move.pass && last_move_is_pass) {
          break;
        }
        last_move_is_pass = move.pass;
      }
      new_game.black_score = players[0].score();
      LOG(debug_log) << std::fixed << std::setprecision(1) << new_game.black_score;
      if (data.size() > maxGames) {
        data.pop_front();
      }
    }
  }

  void store(const std::string& filename) {
    const std::string& ftmp = filename + ".tmp";
    std::ofstream output(ftmp, output.trunc | output.out);
    CHECK(output.is_open()) << "Failed to open " << ftmp << ".";
    output << BOARD_SIZE << " " << std::fixed << std::setprecision(9) << KOMI
           << " " << data.size() << "\n";
    for (const training::Game& game : data) {
      game.store(output);
    }
    output.close();
    std::rename(ftmp.c_str(), filename.c_str());
  }
private:
  const std::string working_dir;
  std::deque<training::Game> data;
};

int main() {
  GenTrainingData gen_training_data("data");
  std::string old_filename = gen_training_data.merge_existing();

  for (size_t i = 0;; ++i) {
    gen_training_data.play(10);

    std::string filename("data/training.");
    filename += std::to_string(std::time(nullptr));
    gen_training_data.store(filename);
    if (!old_filename.empty()) {
      std::experimental::filesystem::remove(old_filename);
    }
    old_filename = filename;
  }
  return 0;
}
