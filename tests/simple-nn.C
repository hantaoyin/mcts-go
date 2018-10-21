// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#include <string>
#include "debug_msg.h"
#include "simple-nn-eval.h"

int main() {
  // This only works when running in the same diretory this source file is in.
  const std::string &filename = network::get_latest("sample_data_dir");
  std::cout << filename << std::endl;
  CHECK(filename == std::string("sample_data_dir/network.1533")) << filename;
  return 0;
}
