// -*- mode:c++; c-basic-offset:2 -*-
// ==================================================================================================
// Change log:
//
// 0.01: File added, implemented basic forward() & store().
#ifndef INCLUDE_GUARD_SIMPLE_NN_EVAL_H__
#define INCLUDE_GUARD_SIMPLE_NN_EVAL_H__

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "debug_msg.h"
#include "board.h"
#include "training.h"

namespace network {

using go_engine::TotalMoves;
static constexpr float eps = 1.e-15f;

class Node;
class Edge {
public:
  Edge(size_t size):v(size),dv(size) {}

  float& operator()(size_t id) {
    return v[id];
  }
  const float& operator()(size_t id) const {
    return v[id];
  }
  float& D(size_t id) {
    return dv[id];
  }
  const float& D(size_t id) const {
    return dv[id];
  }
  size_t size(void) const {
    return v.size();
  }
private:
  std::vector<float> v;
  std::vector<float> dv;
};

class Node {
public:
  Node(Edge &x, Edge &y):x(x), y(y) {
  }
  virtual void forward(void) = 0;
  virtual void backward_propagate(float) = 0;
  virtual void store(std::ofstream&) = 0;
  virtual ~Node() {}
protected:
  Edge &x, &y;
};

class ReLU : public Node {
public:
  ReLU(Edge &x, Edge &y):Node(x, y) {
    ASSERT(x.size() == y.size()) << x.size() << " " << y.size();
  }
  ReLU(std::ifstream& input, Edge &x, Edge &y):Node(x, y) {
    CHECK(x.size() == y.size()) << x.size() << " " << y.size();
    size_t size;
    input >> size;
    CHECK(size == x.size()) << size << " != " << x.size();
    float f;
    input >> f;
    CHECK(std::fabs(f - a) < eps * a) << a << " " << f;
  }

  void forward(void) final {
    for (size_t i = 0; i < x.size(); ++i) {
      y(i) = x(i) > 0 ? x(i) : a * x(i);
    }
  }
  void backward_propagate(float) final {
    for (size_t i = 0; i < x.size(); ++i) {
      x.D(i) = y(i) > 0 ? y.D(i) : a * y.D(i);
    }
  }

  void store(std::ofstream& output) final {
    output << "ReLU: " << x.size() << " " << std::scientific << std::setprecision(9) << a << "\n";
  }
private:
  const float a = 0.01f;
};

class AffineMap : public Node {
public:
  AffineMap(Edge& x, Edge& y)
    : Node(x, y), w(x.size() * y.size()), b(y.size()) {
    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_real_distribution<float> dist;
    for (auto& wi : w) {
      wi = .3f * (dist(e) - 0.5f);
    }
    for (auto& bi : b) {
      bi = .3f * (dist(e) - 0.5f);
    }
  }
  AffineMap(std::ifstream& input, Edge &x, Edge &y)
    : Node(x, y), w(x.size() * y.size()), b(y.size()) {
    size_t x_size, y_size;
    input >> x_size >> y_size;
    CHECK(x_size == x.size()) << x_size << " != " << x.size();
    CHECK(y_size == y.size()) << y_size << " != " << y.size();
    float wd;
    input >> wd;
    CHECK(std::fabs(wd - weight_decay) < eps * weight_decay) << wd << " " << weight_decay;
    for (size_t i = 0; i < w.size(); ++i) {
      input >> w[i];
    }
    for (size_t i = 0; i < b.size(); ++i) {
      input >> b[i];
    }
  }

  void forward(void) final {
    for (size_t i = 0; i < y.size(); ++i) {
      y(i) = b[i];
      for (size_t j = 0; j < x.size(); ++j) {
        y(i) += w[i * x.size() + j] * x(j);
      }
    }
  }
  void backward_propagate(float step_size) final {
    for (size_t j = 0; j < x.size(); ++j) {
      x.D(j) = 0.0f;
    }
    for (size_t i = 0; i < y.size(); ++i) {
      for (size_t j = 0; j < x.size(); ++j) {
        x.D(j) += y.D(i) * w[i * x.size() + j];
        w[i * x.size() + j] = w[i * x.size() + j] * (1.f - 2.f * weight_decay * step_size) - step_size * y.D(i) * x(j);
      }
      b[i] = b[i] * (1.f - 2.f * weight_decay * step_size) - step_size * y.D(i);
    }
  }

  void store(std::ofstream& output) final {
    output << "AffineMap: " << x.size() << " " << y.size() << " ";
    output << std::scientific << std::setprecision(9) << weight_decay << " ";
    // Dump all numbers in w then b into a single line.
    for (float f : w) {
      output << f << " ";
    }
    for (float f : b) {
      output << f << " ";
    }
    output << "\n";
  }
private:
  std::vector<float> w;
  std::vector<float> b;
  static constexpr float weight_decay = 0.002f;
};

class ResidualBlock : public Node {
public:
  ResidualBlock(Edge& x, Edge& y)
    : Node(x, y)
    , e1(x.size()), e2(x.size()), e3(x.size())
    , af1(x, e1)
    , re1(e1, e2)
    , af2(e2, e3)
    , re2(e3, y)
  {
    ASSERT(x.size() == y.size()) << x.size() << " " << y.size();
  }

  ResidualBlock(std::ifstream& input, Edge &x, Edge &y)
    : Node(x, y)
    , e1(x.size()), e2(x.size()), e3(x.size())
    , af1(expect_name(input, "AffineMap:"), x, e1)
    , re1(expect_name(input, "ReLU:"), e1, e2)
    , af2(expect_name(input, "AffineMap:"), e2, e3)
    , re2(expect_name(input, "ReLU:"), e3, y)
  {
    ASSERT(x.size() == y.size()) << x.size() << " " << y.size();
  }

  void forward(void) final {
    af1.forward();
    re1.forward();
    af2.forward();
    for (size_t i = 0; i < x.size(); ++i) {
      e3(i) += x(i);
    }
    re2.forward();
  }
  void backward_propagate(float step_size) final {
    re2.backward_propagate(step_size);
    af2.backward_propagate(step_size);
    re1.backward_propagate(step_size);
    af1.backward_propagate(step_size);
    for (size_t i = 0; i < x.size(); ++i) {
      x.D(i) += e3.D(i);
    }
  }

  void store(std::ofstream& output) final {
    output << "ResidualBlock:\n";
    af1.store(output);
    re1.store(output);
    af2.store(output);
    re2.store(output);
  }
private:
  std::ifstream& expect_name(std::ifstream& input, const std::string& expected) {
    std::string name;
    input >> name;
    CHECK(name == expected) << name << " != " << expected;
    return input;
  }
  Edge e1, e2, e3;
  AffineMap af1;
  ReLU re1;
  AffineMap af2;
  ReLU re2;
};

// This class simply bundles a SoftMax and a single scalar sigmoid (the last element in the output
// edge), used to predict a set of probabilities (summing to 1) and a scalar value.
class SoftMaxAndSigmoid : public Node {
public:
  SoftMaxAndSigmoid(Edge& x, Edge& y)
    : Node(x, y), size(y.size() - 1) {
    ASSERT(y.size() >= 3) << y.size();
    ASSERT(x.size() == y.size()) << x.size() << " " << y.size();
  }
  SoftMaxAndSigmoid(std::ifstream& input, Edge &x, Edge &y)
    : Node(x, y), size(x.size() - 1) {
    CHECK(x.size() == y.size()) << x.size() << " " << y.size();
    size_t size;
    input >> size;
    CHECK(size + 1 == x.size()) << size << " + 1 != " << x.size();
  }

  void forward(void) final {
    // SoftMax
    float xmax = x(0);
    for (size_t i = 1; i < size; ++i) {
      xmax = std::max(xmax, x(i));
    }
    ASSERT(std::fabs(xmax) < 1.e20f) << xmax;

    float sum = 0.;
    for (size_t i = 0; i < size; ++i) {
      y(i) = std::exp(x(i) - xmax);
      sum += y(i);
    }
    ASSERT(sum >= 1.0f && sum <= 1.01f * size) << sum;

    sum = 1.0f / sum;
    for (size_t i = 0; i < size; ++i) {
      y(i) *= sum;
    }

    // Sigmoid
    y(size) = 1.0f / (1.0f + std::exp(-x(size)));
  }

  void backward_propagate(float) final {
    // SoftMax
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
      x.D(i) = y(i) * y.D(i);
      sum += x.D(i);
    }
    for (size_t i = 0; i < size; ++i) {
      x.D(i) -= y(i) * sum;
    }

    // Sigmoid
    x.D(size) = y.D(size) * y(size) * (1.0f - y(size));
  }
  void store(std::ofstream& output) final {
    output << "SoftMaxAndSigmoid: " << size << "\n";
  }
private:
  const size_t size;
};

class MLP {
public:
  MLP(size_t residual_block_size, size_t residual_block_count) {
    const size_t K = residual_block_count;

    // Features are:
    // N * N numbers indicate if the corresponding intersection has a black stone.
    // N * N numbers indicate if the corresponding intersection has a white stone.
    // current player (0 if black, 1 if white).
    e.emplace_back(2 * go_engine::N * go_engine::N + 1);
    e.emplace_back(residual_block_size);
    e.emplace_back(residual_block_size);
    for (size_t i = 0; i < K; ++i) {
      e.emplace_back(residual_block_size);
    }
    // Output layer is a SoftMax function that associate each move with a probability,
    // plus a scalar value that predicts the score.
    e.emplace_back(TotalMoves + 1);
    e.emplace_back(TotalMoves + 1);

    v.emplace_back(std::make_unique<AffineMap>(e[0], e[1]));
    v.emplace_back(std::make_unique<ReLU>(e[1], e[2]));
    for (size_t i = 0; i < K; ++i) {
      v.emplace_back(std::make_unique<ResidualBlock>(e[i + 2], e[i + 3]));
    }
    v.emplace_back(std::make_unique<AffineMap>(e[K + 2], e[K + 3]));
    v.emplace_back(std::make_unique<SoftMaxAndSigmoid>(e[K + 3], e[K + 4]));
  }

  MLP(const std::string& filename) {
    std::ifstream input(filename);
    CHECK(input.is_open()) << "Failed to open " << filename << ".";
    size_t n_edges, n_nodes;
    input >> n_edges >> n_nodes;
    CHECK(n_edges < 1000 && n_nodes < 1000) << n_edges << " " << n_nodes;
    for (size_t i = 0; i < n_edges; ++i) {
      size_t size;
      input >> size;
      e.emplace_back(size);
    }
    CHECK(n_edges == n_nodes + 1 && n_nodes > 0) << filename << " " << n_edges << " " << n_nodes;

    // Now do nodes.
    for (size_t i = 0; i < n_nodes; ++i) {
      std::string type;
      input >> type;
      if (type == "ReLU:") {
        v.emplace_back(std::make_unique<ReLU>(input, e[i], e[i + 1]));
      } else if (type == "AffineMap:") {
        v.emplace_back(std::make_unique<AffineMap>(input, e[i], e[i + 1]));
      } else if (type == "SoftMaxAndSigmoid:") {
        v.emplace_back(std::make_unique<SoftMaxAndSigmoid>(input, e[i], e[i + 1]));
      } else if (type == "ResidualBlock:") {
        v.emplace_back(std::make_unique<ResidualBlock>(input, e[i], e[i + 1]));
      } else {
        CHECK(false) << "Unrecognized node type " << type;
      }
    }
    // CHECK(input.peek() == input.eof()) << filename;
  }

  void store(const std::string& filename) {
    // Line 1 has 2 numbers: # of edges (== e.size()) and # of nodes.
    const std::string& ftmp = filename + ".tmp";
    std::ofstream output(ftmp, output.trunc | output.out);
    CHECK(output.is_open()) << "Failed to open " << ftmp << ".";
    output << e.size() << " " << v.size() << "\n";
    // Line 2 has as many numbers as the # of edges as defined by line
    // 1, each marks the corresponding size.
    for (const Edge& edge : e) {
      output << edge.size() << " ";
    }
    output << "\n";
    // Each subsequent line should fully describe a node.
    for (const auto& node : v) {
      node->store(output);
    }
    output.close();
    std::rename(ftmp.c_str(), filename.c_str());
  }

  const Edge& forward(const go_engine::BoardInfo& b, go_engine::Color next_player) {
    set_input(b, next_player);
    for (size_t i = 0; i < v.size(); ++i) {
      v[i]->forward();
    }
    return e.back();
  }

  void train(const training::Game& game, float step_size, bool debug_log) {
    go_engine::BoardInfo board(KOMI);

    std::array<float, TotalMoves> target_probability;
    for (const training::State& state : game.states) {
      go_engine::Move move = state.move;

      float score = move.color == go_engine::BLACK ? game.black_score > 0 : game.black_score <= 0;
      forward(board, static_cast<go_engine::Color>(move.color));

      float sum = 0.0f;
      for (size_t m = 0; m < TotalMoves; ++m) {
        target_probability[m] = state.count[m] == 0 ? 1.e-5f : state.count[m];
        sum += target_probability[m];
      }
      sum = 1.0f / sum;
      for (size_t m = 0; m < TotalMoves; ++m) {
        target_probability[m] *= sum;
      }

      Edge &eb = e.back();
      const size_t size = eb.size() - 1;
      ASSERT(TotalMoves == size) << size << " != " << TotalMoves;

      // For probabilities, use max likelihood loss function.
      for (size_t m = 0; m < TotalMoves; ++m) {
        eb.D(m) = -target_probability[m] / (1.e-10f + eb(m));
      }
      // For score, we use mean square root as error function.
      eb.D(TotalMoves) = 2. * (eb(TotalMoves) - score);

      // Or this loss function (which proves to be much better than the above since we want the
      // sigmoid to output a near-zero score initially):
    
      // l = (log(z/((1-2e)*s+e)))^2
      // Where e is a very small positive number (to eliminate the singularity), z = sigmoid output, s = score.
      // float s = (1.0f - 2.0f * 1.e-5f) * score + 1.e-5f;
      // eb.D(TotalMoves) = 2.0f * std::log(eb(TotalMoves)/s) / eb(TotalMoves);

      for (size_t i = v.size(); i > 0; --i) {
        v[i - 1]->backward_propagate(step_size);
      }

      if (debug_log) {
        std::cout << board.DebugString() << "\n";
        for (size_t m = 0; m < TotalMoves; ++m) {
          std::cout << std::setw(3) << m << " "
                    << std::setw(11) << state.count[m]
                    << std::setw(11) << std::scientific << std::setprecision(3) << target_probability[m]
                    << std::setw(11) << std::scientific << std::setprecision(3) << eb(m)
                    << std::setw(11) << std::scientific << std::setprecision(3) << eb.D(m) << "\n";
        }
        std::cout << "<score>: " << std::setw(6) << " "
                  << std::setw(11) << std::scientific << std::setprecision(3) << score
                  << std::setw(11) << std::scientific << std::setprecision(3) << eb(TotalMoves)
                  << std::setw(11) << std::scientific << std::setprecision(3) << eb.D(TotalMoves) << "\n\n";
      }

      board.play(move);
    }
    // if (debug_log) {
    //   std::cout << "Using: ";
    //   for (const training::State& state : game.states) {
    //     std::cout << state.move.DebugString() << " ";
    //   }
    //   std::cout << "[" << game.black_score << "]\n";
    // }
  }
private:
  void set_input(const go_engine::BoardInfo& b, go_engine::Color next_player) {
    for (size_t m = 0; m < go_engine::N * go_engine::N; ++m) {
      e[0](m) = b.has_stone(m, next_player);
      e[0](m + go_engine::N * go_engine::N) = b.has_stone(m, go_engine::opposite_color(next_player));
    }
    e[0](2 * go_engine::N * go_engine::N) = next_player;
  }

  std::vector<std::unique_ptr<Node>> v;
  std::vector<Edge> e;
};

// A simple network based on fully connected layers, unsuitbale for large game boards.
class SimpleEvalEngine {
public:
  SimpleEvalEngine()
    : mlp(TotalMoves * 5, 3)
  {}

  SimpleEvalEngine(const std::string& filename)
    : mlp(filename)
  {}

  void store(const std::string& filename) {
    mlp.store(filename);
  }

  float run(const go_engine::BoardInfo&b, go_engine::Color next_player,
            std::array<float, TotalMoves>& prior) {
    const Edge& out = mlp.forward(b, next_player);
    ASSERT(out.size() == TotalMoves + 1) << out.size();
    for (size_t m = 0; m < TotalMoves; ++m) {
      prior[m] = out(m);
    }
    return out(TotalMoves);
  }

  void train(const training::Game& game, float step_size, bool debug_log) {
    mlp.train(game, step_size, debug_log);
  }
private:
  MLP mlp;
};
} // namespace network

#endif  // #ifndef INCLUDE_GUARD_SIMPLE_NN_EVAL_H__
