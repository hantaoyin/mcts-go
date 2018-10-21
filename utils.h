// -*- mode:c++; c-basic-offset:2 -*-
// ==================================================================================================
#ifndef INCLUDE_GUARD_UTILS_H__
#define INCLUDE_GUARD_UTILS_H__

#include <experimental/filesystem>
#include <regex>

#include "config.h"
#include "debug_msg.h"

namespace utils {
inline std::string get_latest(const std::string& path, const std::string& base) {
  std::regex pattern(base + "\\.([0-9]+)");
  std::string ret;
  uint64_t max_ts = 0;
  std::smatch match;
  CHECK(std::experimental::filesystem::is_directory(path)) << path;
  for (const auto &entry : std::experimental::filesystem::directory_iterator(path)) {
    const std::string fname = entry.path().filename();
    if (!std::regex_match(fname, match, pattern)) continue;
    ASSERT(match.size() == 2);
    std::ssub_match timestamp_s = match[1];
    uint64_t timestamp = std::stoull(timestamp_s.str());
    if (timestamp > max_ts) {
      ret = fname;
      max_ts = timestamp;
    }
  }
  if (ret.empty()) {
    return ret;
  } else {
    return path + "/" + ret;
  }
}
}  // utils

#endif  // #ifndef INCLUDE_GUARD_UTILS_H__
