// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
// This main program writes the initial network file, used at the very beginning since all other
// main programs using a network assume that at least one network file exists.
#include <ctime>
#include <string>

#include "simple-nn-eval.h"

int main() {
  std::string filename("data/network.");
  filename += std::to_string(std::time(nullptr));

  network::SimpleEvalEngine eval;
  eval.store(filename);
  return 0;
}
