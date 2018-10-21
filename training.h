// -*- mode:c++; c-basic-offset:2 -*-
// ==================================================================================================
// Change log:
//
// v0.01: File added.
#ifndef INCLUDE_GUARD_TRAINING_H__
#define INCLUDE_GUARD_TRAINING_H__
#include <limits>
#include <vector>

#include "config.h"
#include "board.h"

namespace training {
using go_engine::TotalMoves;

struct State {
  go_engine::Move move;
  std::array<unsigned, go_engine::TotalMoves> count;

  State(std::ifstream& input)
    : move(input)
  {
    for (size_t m = 0; m < go_engine::TotalMoves; ++m) {
      input >> count[m];
    }
  }

  State(go_engine::Move m, const std::array<unsigned, go_engine::TotalMoves>& c)
    : move(m), count(c)
  {}

  void store(std::ofstream& output) const {
    move.store(output);
    output << " ";
    for (size_t m = 0; m < go_engine::TotalMoves; ++m) {
      output << count[m] << " ";
    }
  }
};

struct Game {
  std::vector<State> states;
  float black_score;

  Game(): black_score(std::numeric_limits<float>::quiet_NaN()) {
  }

  Game(std::ifstream& input) {
    size_t size;
    input >> size;
    CHECK(size < 10000) << size;
    states.clear();
    for (size_t i = 0; i < size; ++i) {
      states.emplace_back(input);
    }
    input >> black_score;
  }

  void store(std::ofstream& output) const {
    output << states.size() << "\n";
    for (const State& s : states) {
      s.store(output);
      output << "\n";
    }
    output << std::fixed << std::setprecision(1) << black_score << "\n";
  }
};
}  // namespace training
#endif  // INCLUDE_GUARD_TRAINING_H__
