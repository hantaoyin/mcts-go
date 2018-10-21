// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#include <algorithm>
#include <array>
#include <bitset>
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
#include <unistd.h>
#include <mmintrin.h>

#include "config.h"
#include "debug_msg.h"
#include "board.h"
#include "training.h"
#include "simple-nn-eval.h"
#include "utils.h"

std::vector<training::Game> get_training_data(const std::string& filename) {
  std::cout << "Loading training data from " << filename << "\n";
  std::ifstream input(filename);
  CHECK(input.is_open()) << "Failed to open " << filename << ".";
  size_t board_size;
  float komi;
  size_t n_games;
  input >> board_size >> komi >> n_games;
  CHECK(board_size == BOARD_SIZE) << board_size << " != " << BOARD_SIZE;
  CHECK((komi - KOMI) < 1.e-7f * KOMI) << komi << " != " << KOMI;
  CHECK(n_games < 10000000ULL) << n_games;

  std::vector<training::Game> data;
  for (size_t i = 0; i < n_games; ++i) {
    data.emplace_back(input);
  }
  return data;
}

void try_train(float step_size) {
  const std::string &latest_training = utils::get_latest("data", "training");
  if (latest_training.empty()) {
    return;
  }
  LOG(true) << "Using training file " << latest_training;
  std::vector<training::Game> data(get_training_data(latest_training));
  CHECK(data.size() >= 2) << data.size();

  const std::string &latest_network = utils::get_latest("data", "network");
  CHECK(!latest_network.empty()) << "Can't find a network file.";
  LOG(true) << "Using network file " << latest_network;
  network::SimpleEvalEngine eval(latest_network);

  // We only use the 2nd half of the training data.
  for (size_t i = 0; i < 10000; ++i) {
    size_t half = data.size() / 2;
    size_t id = i % (data.size() - half) + half;
    ASSERT(id < data.size()) << id << " >= " << data.size();
    eval.train(data[id], step_size, i % 500 == 0);
  }

  std::string filename("data/network.");
  filename += std::to_string(std::time(nullptr));
  eval.store(filename);
  std::experimental::filesystem::remove(latest_network);
}

int main() {
  while (true) {
    try_train(0.00001f);
    sleep(1);
  }
  return 0;
}
