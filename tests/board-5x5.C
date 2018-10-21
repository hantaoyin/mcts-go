// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#include <iostream>

#define BOARD_SIZE 5
#include "board.h"

// All tests in this file use a 5x5 board.

//   a b c d e
// 5 . . . . . 5
// 4 . . . . . 4
// 3 . . . . . 3
// 2 . . . . . 2
// 1 . . . . . 1
//   a b c d e

void test1() {
  go_engine::BoardInfo ginfo(0.f);
  std::cout << "Start:\n";
  std::cout << ginfo.DebugString() << "\n\n";

  go_engine::Move moves[] = {
    {go_engine::BLACK, 3, 3},
    {go_engine::WHITE, 3, 2},
    {go_engine::BLACK, 2, 3},
    {go_engine::BLACK, 2, 2},
    {go_engine::BLACK, 1, 3},
    {go_engine::BLACK, 3, 1},
    {go_engine::BLACK, 4, 2},
  };

  unsigned expected_liberty_count[] = {
    4, 3, 5, 6, 7, 3, 3,
  };

  for (size_t i = 0; i < sizeof(moves) / sizeof(go_engine::Move); ++i) {
    std::cout << moves[i].DebugString() << "\n";
    ginfo.play(moves[i]);
    unsigned liberty_count = ginfo.count_liberty(moves[i].loc).first;
    std::cout << ginfo.DebugString() << "\n\n";
    CHECK(liberty_count == expected_liberty_count[i]) << liberty_count << " " << expected_liberty_count[i];
  }
  CHECK(ginfo.is_valid({go_engine::BLACK, 3, 2}) == true);
  CHECK(ginfo.is_valid({go_engine::WHITE, 3, 2}) == false);
}

void test2() {
  go_engine::BoardInfo ginfo("X X X . ."
                             "X O . O ."
                             "X X X . ."
                             ". . . . ."
                             ". . . . .", 0.f);
  go_engine::Move move(go_engine::BLACK, 3, 2);
  CHECK(ginfo.is_valid(move) == true);
  ginfo.play(move);
  unsigned liberty_count = ginfo.count_liberty(move.loc).first;
  std::cout << ginfo.DebugString() << "\n\n";
  CHECK(liberty_count == 6) << liberty_count;
}

void test3() {
  go_engine::BoardInfo ginfo(". X X . ."
                             "X O . O ."
                             "X X X . ."
                             ". . . . ."
                             ". . . . .", 0.f);
  go_engine::Move move(go_engine::BLACK, 3, 2);
  CHECK(ginfo.is_valid(move) == true);
  ginfo.play(move);
  unsigned liberty_count = ginfo.count_liberty(move.loc).first;
  std::cout << ginfo.DebugString() << "\n\n";
  CHECK(liberty_count == 7) << liberty_count;
}

void test4() {
  go_engine::BoardInfo ginfo(". X X X ."
                             "X O . O X"
                             "X X X X ."
                             ". . . . ."
                             ". . . . .", 0.f);
  {
    go_engine::Move move(go_engine::WHITE, 3, 2);
    CHECK(ginfo.is_valid(move) == false);
  }
  {
    go_engine::Move move(go_engine::BLACK, 3, 2);
    CHECK(ginfo.is_valid(move) == true);
    ginfo.play(move);
    unsigned liberty_count = ginfo.count_liberty(move.loc).first;
    std::cout << ginfo.DebugString() << "\n\n";
    CHECK(liberty_count == 9) << liberty_count;
  }
}

// Test superko implementation.
void test5() {
  go_engine::BoardInfo ginfo(". . O X ."
                             ". O . O X"
                             ". . O X ."
                             ". . . . ."
                             ". . . . .", 0.f);
  {
    go_engine::Move move(go_engine::BLACK, 3, 2);
    std::cout << ginfo.DebugString() << "\n\n";
    CHECK(ginfo.is_valid(move) == true);
    ginfo.play(move);
  }
  {
    go_engine::Move move(go_engine::WHITE, 3, 3);
    CHECK(ginfo.is_valid(move) == false);
    std::cout << ginfo.DebugString() << "\n\n";
  }
}

// Test superko implementation.
void test6() {
  go_engine::BoardInfo ginfo(". . . . ."
                             "O O X X X"
                             ". . O . ."
                             "O O X X X"
                             ". . . . .", 0.f);
  go_engine::Move moves[] = {
    {go_engine::WHITE, 2, 3},
    {go_engine::WHITE, 2, 4},
    {go_engine::BLACK, 2, 0},
    {go_engine::BLACK, 2, 1},
  };
  for (const auto& move : moves) {
    ginfo.play(move);
  }
  std::cout << ginfo.DebugString() << "\n\n";
  CHECK(ginfo.is_valid({go_engine::WHITE, 2, 2}) == false);
}

// This shows that we are implementing positional superko rule.  If situational superko rule is
// used, then the last move in this test is legal.
void test7() {
  go_engine::BoardInfo ginfo(". . . . ."
                             "O X X . ."
                             ". O . X ."
                             "O X X . ."
                             ". . . . .", 0.f);
  go_engine::Move moves[] = {
    {go_engine::WHITE, 2, 2},
    {go_engine::BLACK, 2, 0},
  };
  for (const auto& move : moves) {
    ginfo.play(move);
  }
  std::cout << ginfo.DebugString() << "\n\n";
  CHECK(ginfo.is_valid({go_engine::WHITE, 2, 1}) == false);
}

// This tests the is_valid() implementation that checks the superko rule.  Here, after white's move,
// black can't play at a3 (row 2, column 0) because it would reproduce a configuration seen before.
// This test checks if this is implemented correctly.  The tricky part is that black's move at a3
// has a white group with a single liberty adjacent to it from 2 sides: a correct implementation
// must not double count this group (when computing the Zobrist hash, for example).
void test8() {
  go_engine::BoardInfo ginfo(". . . . ."
                             "O X X . ."
                             "X . X . ."
                             ". . X . ."
                             "X X . . .", 0.f);
  go_engine::Move moves[] = {
    {go_engine::WHITE, 1, 0},
    {go_engine::WHITE, 1, 1},
    {go_engine::WHITE, 2, 1},
  };
  for (const auto& move : moves) {
    ginfo.play(move);
  }
  std::cout << ginfo.DebugString() << "\n\n";
  CHECK(ginfo.is_valid({go_engine::BLACK, 2, 0}) == false);
}

// Same as above, but here a white group with a single liberty is adjacent to a proposed black's
// move from all 4 sides.
void test9() {
  go_engine::BoardInfo ginfo(". X X X ."
                             "X . . . X"
                             "X . X . X"
                             "X . . . X"
                             ". X X X .", 0.f);
  go_engine::Move moves[] = {
    {go_engine::WHITE, 1, 1},
    {go_engine::WHITE, 1, 2},
    {go_engine::WHITE, 1, 3},
    {go_engine::WHITE, 2, 1},
    {go_engine::WHITE, 2, 3},
    {go_engine::WHITE, 3, 1},
    {go_engine::WHITE, 3, 2},
    {go_engine::WHITE, 3, 3},
  };
  for (const auto& move : moves) {
    ginfo.play(move);
  }
  std::cout << ginfo.DebugString() << "\n\n";
  CHECK(ginfo.is_valid({go_engine::BLACK, 2, 2}) == false);
}

// Test scoring function.
void test10() {
  {
    go_engine::BoardInfo ginfo(2.);
    CHECK(ginfo.score() == -2.);
  }
  {
    go_engine::BoardInfo ginfo(". . . . ."
                               "O O O O O"
                               ". O . X ."
                               "X X X X X"
                               ". . . . .", 2.);
    CHECK(ginfo.score() == -2.);

    ginfo.play({go_engine::WHITE, 2, 0});
    CHECK(ginfo.score() == -3.) << ginfo.score();

    ginfo.play({go_engine::WHITE, 2, 4});
    CHECK(ginfo.score() == -4.) << ginfo.score();

    ginfo.play({go_engine::BLACK, 0, 2});
    CHECK(ginfo.score() == -4.) << ginfo.score();
  }
  {
    go_engine::BoardInfo ginfo(". X O . O"
                               "O O O O ."
                               ". O X X ."
                               ". X X X X"
                               ". . . X .", 2.);
    CHECK(ginfo.score() == -1.) << ginfo.score();
  }
  {
    go_engine::BoardInfo ginfo(". X . . O"
                               "O O O O ."
                               ". O . O ."
                               ". X O O X"
                               ". . X X .", 2.);
    CHECK(ginfo.score() == -6.) << ginfo.score();
  }
  {
    go_engine::BoardInfo ginfo(". X X X ."
                               "X . O O X"
                               "X O . O X"
                               "X O O . X"
                               ". X X X .", 2.);
    CHECK(ginfo.score() == 7.) << ginfo.score();
  }
}

int main() {
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  test8();
  test9();
  test10();
  return 0;
}
