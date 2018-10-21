// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#include <string>
#include "debug_msg.h"

int main(int argc, const char const* argv[]) {
  CHECK(argc == 2) << "Usage: " << argv[0] << " <true|false>";
  std::string value(argv[1]);
  ASSERT(value == "true") << value;
  return 0;
}
