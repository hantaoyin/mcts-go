// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#ifndef INCLUDE_GUARD_MCTS_H__
#define INCLUDE_GUARD_MCTS_H__

#include <array>
#include <functional>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <utility>
#include <vector>

#include "config.h"
#include "debug_msg.h"
#include "board.h"

namespace mcts {

using go_engine::TotalMoves;

struct Node {
  // TODO: Some fields can be lazily allocated to save memory.
  std::array<float, TotalMoves> prior;
  std::array<unsigned, TotalMoves> count;
  std::array<float, TotalMoves> value;
  std::array<unsigned, TotalMoves> child;
  unsigned total_count;
  // score from value network.
  float prior_score;
};

template<size_t N>
class DirichletDist {
public:
  DirichletDist(float c)
    : e(rd())
    , gamma(c, 1.03) {}

  const std::array<float, N>& gen() {
    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      x[i] = gamma(e);
      ASSERT(!std::isnan(x[i]));
      sum += x[i];
    }
    sum = 1.0f / sum;
    for (size_t i = 0; i < N; ++i) {
      x[i] *= sum;
    }
    return x;
  }
private:
  std::array<float, N> x;

  std::random_device rd;
  std::default_random_engine e;
  std::gamma_distribution<float> gamma;
};

// This eval engine simply sets all priors to equal values and score to 0.5 (both players have equal
// probability to win).
struct DummyEvalEngine {
  float run(const go_engine::BoardInfo&, go_engine::Color, std::array<float, TotalMoves>& prior) {
    for (size_t m = 0; m < TotalMoves; ++m) {
      prior[m] = 1.0f / static_cast<float>(TotalMoves);
    }
    return 0.5f;
  }
};

template<typename EvalEngine>
class Tree {
  static constexpr unsigned Unexplored = static_cast<unsigned>(-1);
public:
  template<typename... Args>
  Tree(float komi, go_engine::Color c, Args&&... args)
    :board(komi), color(c), id(0)
    , eval(std::forward<Args>(args)...)
    , engine(std::random_device()())
    , dir(1.03f)
  {
    init_node(board);
  }

  void reset() {
    board.reset();
    id = 0;
    states.clear();
    history.clear();
    init_node(board);
  }

  const std::array<unsigned, go_engine::TotalMoves>& get_search_count() const {
    return states[id].count;
  }

  go_engine::Move gen_play(bool debug_log) {
    CHECK(board.get_next_player() == color);
    for (size_t i = 0; i < SearchCount; ++i) {
      search_from(id, !history.empty() && history.back().pass, false);
    }

    float sum = 0.0f;
    const Node& node = states[id];
    std::array<float, TotalMoves> p;
    const float inv_temp = history.size() < go_engine::N ? 1.0f : 5.0f;
    for (size_t m = 0; m < TotalMoves; ++m) {
      if (node.prior[m] < 0.0f) {
        continue;  // Invalid moves.
      }
      p[m] = std::pow(node.count[m], inv_temp);
      sum += p[m];
    }

    LOG(debug_log) << board.DebugString();
    if (debug_log) {
      for (size_t m = 0; m < TotalMoves; ++m) {
        if (node.prior[m] < 0.0f) continue;  // Invalid moves.
        go_engine::Move move(color, m);
        LOG(debug_log)
          << "    " << move.DebugString()
          << ": prior = " << std::fixed << std::setprecision(4) << std::setfill(' ') << node.prior[m]
          << ", count = " << std::setw(6) << std::setfill(' ') << node.count[m]
          << ", value = " << std::fixed << std::setprecision(4) << std::setfill(' ')
          << (node.count[m] == 0 ? 0.5f : node.value[m] / node.count[m]);
      }
    }
    LOG(debug_log) << "    <est. score>: " << std::fixed << std::setprecision(4) << std::setfill(' ')
                   << node.prior_score;

    // Since pass is always a valid move, sum should always be positive.
    ASSERT(sum > 0);
    float r = dist(engine) * sum;
    for (size_t m = 0; m < TotalMoves; ++m) {
      if (node.prior[m] < 0.0f) continue;
      r -= p[m];
      if (r < 0.0f) {
        go_engine::Move move(color, m);
        LOG(debug_log) << "(MCTS)==> play: " << move.DebugString() << "\n";
        return move;
      }
    }
    CHECK(false) << board.DebugString();
    return {color};
  }

  void play(go_engine::Move move) {
    ASSERT(board.is_valid(move)) << board.DebugString();
    board.play(move);
    history.push_back(move);

    size_t m = move.id();
    ASSERT(m < go_engine::TotalMoves) << move.DebugString();
    auto& node = states[id];
    if (node.child[m] == Unexplored) {
      node.child[m] = states.size();
      init_node(board);
    }
    id = node.child[m];
  }

  float score() const {
    return color == go_engine::BLACK ? board.score() : -board.score();
  }
private:
  // Perform a full Monte Carlo tree search from state id.  Return value is the score of this move
  // (winning probability of the current player).
  float search_from(size_t root, bool last_move_is_pass, bool debug_log) {
    // TODO: Very expensive.
    go_engine::BoardInfo local_board(board);
    std::function<unsigned(go_engine::Color, size_t, bool)> search_recursively =
      [this, &search_recursively, &local_board, debug_log](go_engine::Color c, size_t root, bool last_move_is_pass) -> float {
      ASSERT(root < states.size());
      auto& node = states[root];
      LOG(debug_log) << "\n" << local_board.DebugString();

      float ucb1_max = -std::numeric_limits<float>::infinity();
      unsigned m_max = TotalMoves;
      const float nsq = sqrt((float)node.total_count);
      for (unsigned m = 0; m < TotalMoves; ++m) {
        go_engine::Move move(c, m);
        if (node.prior[m] < 0.0f) {
          continue;
        }
        if (local_board.is_valid(move)) {
          float u = node.count[m] == 0 ? 0.5f : node.value[m] / static_cast<float>(node.count[m]);
          u += node.prior[m] * nsq / (1 + node.count[m]);
          LOG(debug_log) << "    " << move.DebugString() << " ==> prior = " << std::setfill('0') << std::fixed << node.prior[m]
                         << ", visit = " << std::setw(10) << std::setfill(' ') << node.count[m]
                         << ", value = " << std::setprecision(3) << std::setfill(' ') << std::scientific
                         << (node.count[m] == 0 ? 0.5 : node.value[m] / node.count[m])
                         << ", ucb = " << std::setw(14) << std::setfill(' ') << std::scientific << u;
          if (u > ucb1_max) {
            ucb1_max = u;
            m_max = m;
          }
        } else {
          node.prior[m] = -1.0f;
        }
      }
      // Note that pass is always a valid move.
      ASSERT(m_max < TotalMoves) << "\n" << local_board.DebugString();

      go_engine::Move move(c, m_max);
      LOG(debug_log) << "(MCTS)==> Move: " << move.DebugString();
      local_board.play(move);

      float score;
      if (move.pass && last_move_is_pass) {  // 2 consecutive passes, end the game now.
        score = c == go_engine::BLACK ? local_board.score() >= 0 : local_board.score() < 0;
        LOG(debug_log) << "(MCTS)==> " << go_engine::to_string(c) << ": score (Count) = " << score;
      } else if (node.child[m_max] == Unexplored) {
        node.child[m_max] = states.size();
        score = 1.0f - init_node(local_board);
        LOG(debug_log) << "(MCTS)==> " << go_engine::to_string(c) << ": score (NN) = " << score;
      } else {
        ASSERT(node.child[m_max] < states.size());
        // Continue by recursive play.
        score = 1.0f - search_recursively(go_engine::opposite_color(c), node.child[m_max], move.pass);
      }
      // Update.
      ++states[root].count[m_max];
      states[root].value[m_max] += score;
      ++states[root].total_count;
      return score;
    };
    return search_recursively(board.get_next_player(), root, last_move_is_pass);
  }

  // The new node is always appended to the end of the vector of nodes, which means its id (pointer)
  // is implicitly defined.
  float init_node(const go_engine::BoardInfo& b) {
    states.emplace_back();
    auto& node = states.back();
    for (size_t m = 0; m < TotalMoves; ++m) {
      node.count[m] = 0;
      node.value[m] = 0.0f;
      node.child[m] = Unexplored;
    }
    node.total_count = 0;
    node.prior_score = eval.run(b, b.get_next_player(), node.prior);
    // Add Dirichlet noise to encourage exploration.
    const std::array<float, TotalMoves>& noise = dir.gen();
    for (size_t m = 0; m < TotalMoves; ++m) {
      node.prior[m] = node.prior[m] * 0.75f + noise[m] * 0.25f;
    }
    return node.prior_score;
  }

  // Control parameters.
  static constexpr size_t SearchCount = 1000;

  go_engine::BoardInfo board;
  const go_engine::Color color;
  size_t id; // Current Node in states corresponding to the board.
  EvalEngine eval;

  std::vector<Node> states;
  std::vector<go_engine::Move> history;
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist;
  DirichletDist<TotalMoves> dir;
};
}  // mcts

#endif  // #ifndef INCLUDE_GUARD_MCTS_H__
