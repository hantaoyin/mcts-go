// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#ifndef INCLUDE_GUARD_DEBUG_MSG_H__
#define INCLUDE_GUARD_DEBUG_MSG_H__

#include<iostream>

namespace check_impl {
// For delayed invocation of exit.
struct Terminate {
  ~Terminate() {
    std::cerr << "\n";
    std::exit(-1);
  }
};

struct NewLine {
  ~NewLine() {
    std::cout << "\n";
  }
};
}  // namespace check_impl

// This is meant to be a check always in place.
//
// Usage:
//
// Check if an integer a is even:
//
// CHECK(a % 2 == 0) << a << "\n";
#define CHECK(condition) __builtin_expect((condition), 1) ? \
  std::cerr :                                               \
  (check_impl::Terminate(), std::cerr)                      \
  << "Condition `" #condition "` failed in "                \
  << __FILE__ << " line " << __LINE__ << ", msg = "

// This is meant to be a check only in place in debug mode.  Unlike CHECK(condition),
// ASSERT(condition) doesn't evaluate condition in NDEBUG mode.
//
// About NDEBUG: some compiler toolchain / setup automatically add NDEBUG such as the one Python
// uses to compile its modules, hence the same name is avoided here.
#ifdef MY_NDEBUG
#  define ASSERT(condition) true || std::cerr
#else
#  define ASSERT(condition) CHECK((condition))
#endif  // MY_NDEBUG

// If condition is true, then LOG(condition) behaves like std::cout, otherwise does nothing and
// discards any following << ... construction without evaluating them.
//
// This is meant to reduce the frequency of logging messages, so condition is expected to be false
// most of the time.
#define LOG(condition) __builtin_expect(!(condition), 1) ? \
  std::cout : (check_impl::NewLine(), std::cout)

#endif  // #ifndef INCLUDE_GUARD_DEBUG_MSG_H__
