// -*- mode:c++; c-basic-offset:2; coding:utf-8-unix -*-
// ==================================================================================================
#ifndef INCLUDE_GUARD_BOARD_H__
#define INCLUDE_GUARD_BOARD_H__

#include <array>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "config.h"
#include "debug_msg.h"

namespace go_engine {

#ifndef BOARD_SIZE
#error "Must define BOARD_SIZE."
#endif

// The size of the game board.
static constexpr unsigned N = BOARD_SIZE;
// We have data structures depending on N being small enough.
static_assert(N <= 19);

constexpr size_t TotalMoves = go_engine::N * go_engine::N + 1;

enum Color {
  BLACK = 0,
  WHITE = 1,
};

inline const char* to_string(Color c) {
  return c == BLACK ? "Black" : "White";
}

inline Color opposite_color(Color c) {
  return static_cast<Color>(1U - c);
}

struct Move {
  Move(Color c, unsigned id = N * N)
    : color(c)
    , loc(id)
    , pass(id == N * N)
  {
    ASSERT(id <= N * N);
  }

  Move(Color c, unsigned row, unsigned col)
    : color(c)
    , loc(row * N + col)
    , pass(0)
  {
    ASSERT(row < N && col < N) << "(" << row << ", " << col << ")\n";
  }


  Move(std::ifstream& input) {
    std::string s;
    input >> s;
    CHECK(s.size() == 4 || s.size() == 6) << s;
    CHECK(s[0] == 'B' || s[0] =='W') << s;
    color = s[0] == 'B' ? 0 : 1;
    CHECK(s[1] == ':') << s;
    pass = s.size() == 6;
    if (s.size() == 6) {
      loc = N * N;
    } else {  // size == 4
      unsigned row = s[2] - 'a';
      unsigned col = s[3] - 'a';
      unsigned l = row * N + col;
      CHECK(l < N * N) << s;
      loc = l;
    }
  }

  void store(std::ofstream& output) const {
    output << (color == BLACK ? "B:" : "W:");
    if (pass) {
      output << "pass";
    } else {
      char row = 'a' + loc / N;
      char col = 'a' + loc % N;
      output << row << col;
    }
  }

  std::string DebugString() const {
    unsigned short row = loc / N;
    unsigned short col = loc % N;
    std::stringstream ss;
    ss << (color == BLACK ? "B:" : "W:");
    if (pass) {
      ss << "pass";
    } else {
      ss << (char)(col + 'a') << std::setw(3) << std::left << row + 1;
    }
    return ss.str();
  }

  unsigned id() const {
    return loc;
  }

  unsigned row() const {
    return loc / N;
  }
  unsigned col() const {
    return loc % N;
  }

  unsigned short color : 1;
  unsigned short loc : 9;
  unsigned short pass : 1;
};

class ZobristHash {
public:
  using type = uint64_t;

  ZobristHash() {
    // std::random_device rd;
    // std::default_random_engine engine(rd());
    std::default_random_engine engine(100);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < seed.size(); ++i) {
      seed[i] = dist(engine);
    }
  }

  type hash(unsigned loc, Color c) const {
    return seed[loc + (c == BLACK ? 0 : N * N)];
  }
private:
  std::array<uint64_t, N * N * 2> seed;
};

extern ZobristHash zobrist_hash;
using ZobristHashType = typename ZobristHash::type;

// This class is used to track:
// 1. If 2 stones belong to the same group.
// 2. The liberty count of each group.
class BoardInfo {
public:
  // Default constructor creates a state matching the beginning of the game.  komi is always added
  // to white player.
  explicit BoardInfo(float _komi)
    : komi(_komi)
    , existing_states(nullptr) {}

  explicit BoardInfo(const BoardInfo& b)
    : board(b.board)
    , unique_id(b.unique_id)
    , hash(b.hash)
    , komi(b.komi)
    , existing_states(&b.seen_states)
  {
    CHECK(b.existing_states == nullptr) << "Can't duplicate from an already duplicated board.";
  }

  // Construct from a string representing the board.  This is mainly for debugging purposes.
  //
  // Input string is interpreted as a row major representation of the board, where:
  // 1. . (dot) represents an empty intersection.
  // 2. X represents a black stone.
  // 3. O represents a white stone.
  // 4. Whitespaces are ignored.
  explicit BoardInfo(const std::string& input, float _komi)
    : komi(_komi)
    , existing_states(nullptr) {
    std::string s;
    for (char c : input) {
      if (c == ' ') continue;
      s.push_back(c);
    }
    CHECK(s.size() == N * N) << "Invalid board: " << s;
    for (size_t row = 0; row < N; ++row) {
      for (size_t col = 0; col < N; ++col) {
        size_t loc = (N - 1 - row) * N + col;
        if (s[loc] == '.') continue;
        CHECK(s[loc] == 'X' || s[loc] == 'O') << "Invalid board: " << s;
        Move move(s[loc] == 'X' ? BLACK : WHITE, row * N + col);
        CHECK(is_valid(move)) << move.DebugString();
        play(move);
      }
    }
  }

  // Reset the board to a state matching the beginning of the game.  komi is not changed, although
  // this is not a hard requirement.
  void reset() {
    memset(&board, 0, sizeof(board));
    unique_id = 0;
    hash = 0;
    seen_states.clear();
  }

  std::string DebugString() const {
    std::stringstream ss;
    auto print_row_labels = [&ss]() {
      ss << "  ";
      for (size_t col = 0; col < N; ++col) {
        ss << std::setw(2) << (char)('a' + col);
      }
      ss << "   ";
      for (size_t col = 0; col < N; ++col) {
        ss << std::setw(2) << (char)('a' + col);
      }
    };
    print_row_labels();
    ss << "\n";
    for (size_t row = N - 1; row != static_cast<size_t>(-1); --row) {
      ss << std::setw(2) << row + 1;
      for (size_t col = 0; col < N; ++col) {
        unsigned loc = row * N + col;
        const Point& point = board[loc];
        const char v = point.has_stone ? (point.color == BLACK ? 'X' : 'O') : '.';
        ss << std::setw(2) << v;
      }
      ss << std::setw(3) << row + 1;
      for (size_t col = 0; col < N; ++col) {
        unsigned loc = row * N + col;
        if (board[loc].has_stone) {
          unsigned lc = count_liberty(loc).first;
          unsigned char v;
          if (lc >= 36) v = '+';
          else if (lc >= 10) v = 'A' + lc;
          else v = '0' + lc;
          ss << std::setw(2) << v;
        } else {
          ss << std::setw(2) << '.';
        }
      }
      ss << std::setw(3) << row + 1 << "\n";
    }
    print_row_labels();
    ss << "\nHash: " << std::hex << std::setfill('0') << hash << std::setfill(' ');
    return ss.str();
  }

  // Black's score - White's score.  So if score > 0, then black wins the game.  Use Tromp-Taylor
  // rules.
  float score() const {
    unsigned count[2] = {0, 0};

    const unsigned short mark = next_id();
    // Return value is: color, count.
    //
    // Color == 0: This only occurs if the entire board is empty.
    // Color == 1: The empty group containing loc is surrounded by black stones.  
    // Color == 2: The empty group containing loc is surrounded by white stones.
    // Color == 3: The empty group is not surrounded by a single type of stone.
    std::function<std::pair<unsigned, unsigned>(unsigned)> check_empty =
      [this, &check_empty, mark](unsigned loc) -> std::pair<unsigned, unsigned> {
      ASSERT(loc < N * N) << loc;
      if (board[loc].has_stone) {
        return std::make_pair(board[loc].color == BLACK ? 1U : 2U, 0U);
      }
      if (board[loc].payload == mark) {
        return std::make_pair(0U, 0U);
      }
      // Empty.
      board[loc].payload = mark;
      unsigned y = loc % N;

      unsigned color = 0;
      unsigned count = 1;
      if (loc >= N) {
        auto result = check_empty(loc - N);
        color |= result.first;
        count += result.second;
      }
      if (loc < N * (N - 1)) {
        auto result = check_empty(loc + N);
        color |= result.first;
        count += result.second;
      }
      if (y > 0) {
        auto result = check_empty(loc - 1);
        color |= result.first;
        count += result.second;
      }
      if (y + 1 < N) {
        auto result = check_empty(loc + 1);
        color |= result.first;
        count += result.second;
      }
      return std::make_pair(color, count);
    };
    for (unsigned loc = 0; loc < N * N; ++loc) {
      if (board[loc].has_stone) {
        ++count[board[loc].color];
      } else if (board[loc].payload != mark) {
        auto result = check_empty(loc);
        ASSERT(result.first <= 3) << result.first << " " << result.second;
        if (result.first == 1 || result.first == 2) {
          count[result.first - 1] += result.second;
        }
      }
    }
    return (float)(count[BLACK]) - (float)(count[WHITE]) - komi;
  }

  // Count liberty of the group as well as the Zobrist hash of the group containing the stone at
  // loc.
  //
  // Assumes the input location has a stone.
  std::pair<unsigned, ZobristHashType> count_liberty(const unsigned loc) const {
    ASSERT(loc < N * N) << loc;
    ASSERT(board[loc].has_stone) << DebugString();

    const unsigned short mark = next_id();
    auto has_liberty = [this, mark](unsigned l) -> unsigned {
      ASSERT(l < N * N);
      if (board[l].has_stone || board[l].payload == mark) {
        return 0;
      }
      board[l].payload = mark;
      return 1;
    };

    unsigned p = loc;
    unsigned count = 0;
    ZobristHashType h = 0;
    const Color c = static_cast<go_engine::Color>(board[loc].color);
    do {
      unsigned y = p % N;
      h ^= zobrist_hash.hash(p, c);
      if (p >= N)          count += has_liberty(p - N);
      if (p < N * (N - 1)) count += has_liberty(p + N);
      if (y > 0)           count += has_liberty(p - 1);
      if (y + 1 < N)       count += has_liberty(p + 1);

      ASSERT(p < N * N);
      p = board[p].payload;
      ASSERT(board[p].has_stone && board[p].color == c) << loc << " " << p << "\n" << DebugString();
    } while (p != loc);
    return std::make_pair(count, h);
  }

  // Check if this move is valid (i.e., it's not a suicide move and it doesn't violate the ko rule).
  //
  // Algorithm used (sans the ko rule part):
  //
  // For all 4 adjacent locations:
  //   If location is empty  -> return valid
  //   If location has a group of the opposite color and it has only 1 liberty -> return valid
  //   If location has a group of the same color and it has more than 1 liberty -> return valid
  // return invalid.
  bool is_valid(Move move) const {
    if (move.pass) return true;
    if (board[move.loc].has_stone) return false;

    // maybe_valid == true <==> This move is valid except that it still needs to pass the superko
    // check.
    bool maybe_valid = false;

    ZobristHashType h = zobrist_hash.hash(move.loc, move.color == BLACK ? BLACK : WHITE);
    std::array<ZobristHashType, 4> removed_group_hash;
    size_t k = 0;
    auto valid = [this, color=move.color, &removed_group_hash, &k](unsigned loc) -> bool {
      ASSERT(loc < N * N);
      if (!board[loc].has_stone) {
        return true;
      }
      auto v = count_liberty(loc);
      ASSERT(v.first > 0) << loc << "\n" << DebugString();
      if (board[loc].color == color) {
        return v.first > 1;
      } else {
        if (v.first == 1) {
          removed_group_hash[k++] = v.second;
        }
        return v.first == 1;
      }
    };

    unsigned y = move.loc % N;
    if (move.loc >= N && valid(move.loc - N))          maybe_valid = true;
    if (move.loc < N * (N - 1) && valid(move.loc + N)) maybe_valid = true;
    if (y > 0 && valid(move.loc - 1))                  maybe_valid = true;
    if (y + 1 < N && valid(move.loc + 1))              maybe_valid = true;

    // Crude method to avoid computing the hash of the same group twice.
    for (size_t i = 0; i < k; ++i) {
      h ^= removed_group_hash[i];
      for (size_t j = 0; j < i; ++j) {
        if (removed_group_hash[i] == removed_group_hash[j]) {
          h ^= removed_group_hash[i];
          break;
        }
      }
    }

    if (__builtin_expect(maybe_valid, 1)) {
      // This implements the superko rule.  Since the hash doesn't record whose turn it is, this
      // effectively implements the positional superko rule, which forbids recreation of any
      // previously seen board configurations, regardless of whose turn it is.
      if (existing_states != nullptr &&
          existing_states->find(hash ^ h) != existing_states->end()) {
        return false;
      }
      return seen_states.find(hash ^ h) == seen_states.end();
    } else {
      return false;
    }
  }

  void play(Move move) {
    if (move.pass) {
      return;
    }
    ASSERT(is_valid(move)) << move.DebugString();
    ASSERT(move.loc < N * N);
    Point& point = board[move.loc];
    hash ^= zobrist_hash.hash(move.loc, move.color == BLACK ? BLACK : WHITE);

    point.has_stone = true;
    point.color = move.color;
    point.payload = move.loc;

    // 1. Combine this stone and its adjacent stones of same color into one group.
    auto combine_same_color = [this, loc=move.loc](unsigned adj) {
      ASSERT(adj < N * N);
      if (board[adj].has_stone && board[adj].color == board[loc].color && !same_group(loc, adj)) {
        unsigned short t = board[loc].payload;
        board[loc].payload = board[adj].payload;
        board[adj].payload = t;
      }
    };
    unsigned short col = move.loc % N;
    if (move.loc >= N)          combine_same_color(move.loc - N);
    if (move.loc < N * (N - 1)) combine_same_color(move.loc + N);
    if (col > 0)                combine_same_color(move.loc - 1);
    if (col + 1U < N)           combine_same_color(move.loc + 1);

    // 3. For each adjacent *group* of opposite color, remove it if necessary.
    auto update_opp = [this, loc=move.loc](unsigned l) {
      ASSERT(l < N * N);
      if (board[l].has_stone && board[l].color != board[loc].color) {
        unsigned lc = count_liberty(l).first;
        if (lc == 0) {
          hash ^= remove_group(l);
        }
      }
    };


    if (move.loc >= N)          update_opp(move.loc - N);
    if (move.loc < N * (N - 1)) update_opp(move.loc + N);
    if (col > 0)                update_opp(move.loc - 1);
    if (col + 1U < N)           update_opp(move.loc + 1);
    ASSERT(seen_states.find(hash) == seen_states.end())
      << move.DebugString() << "\n" << DebugString() << std::hex << hash;
    // ASSERT(seen_states.size() < 2 * N * N) << seen_states.size();
    seen_states.insert(hash);
  }

  bool has_stone(unsigned loc, Color c) const {
    ASSERT(loc < N * N) << loc;
    return board[loc].has_stone && board[loc].color == c;
  }
private:
  ZobristHashType remove_group(const unsigned loc) {
    ASSERT(loc < N * N);
    ASSERT(board[loc].has_stone) << loc << "\n" << DebugString();
    Color c = board[loc].color == BLACK ? BLACK : WHITE;
    unsigned p = loc;
    ZobristHashType h = 0;
    while(true) {
      ASSERT(p < N * N);
      unsigned next = board[p].payload;
      memset(&board[p], 0, sizeof(Point));
      h ^= zobrist_hash.hash(p, c);
      p = next;
      if (p == loc) {
        break;
      }
      ASSERT(board[p].has_stone && board[p].color == c) << loc << " " << p << "\n" << DebugString();
    }
    return h;
  }

  struct Point {
    unsigned short has_stone : 1;
    unsigned short color : 1;

    // 1. For locations where there is a stone, this field is used to construct a circular linked list,
    // i.e., its value is the index of the next stone in the linked list.
    //
    // 2. For empty locations, payload is used as a scratch space.  So far this is always used as a
    // marker indicating if this location has been visited before.
    //
    // The way we count liberties: pick up a unique number that's guaranteed to be not equal to any
    // payload field of any currently unoccupied locations.  When we go over the linked list, we
    // mark any adjacent empty location using this unique number, this provides a way to avoid
    // duplicated counting.
    mutable unsigned short payload : 14;
  };

  // Check if 2 stones belong to the same group.
  bool same_group(unsigned la, unsigned lb) const {
    ASSERT(la < N * N && lb < N * N);
    ASSERT(board[la].has_stone && board[lb].has_stone);
    if (board[la].color != board[lb].color) return false;

    unsigned p = la;
    do {
      ASSERT(p < N * N);
      if (p == lb) return true;
      p = board[p].payload;
      ASSERT(board[p].has_stone && board[p].color == board[la].color);
    } while (p != la);
    return false;
  }

  // Choose a unique ID for functions requiring a marker on empty locations such as count_liberty(),
  // score().
  //
  // We keep an internal value, add 1 to it, and return the updated value every time we need a new
  // unique ID.  This ensures that the returned value is never seen before.  When this internal
  // value wraps around, we reset the payload field of all empty locations on the board (expensive).
  //
  // Obvious, this method is not thread safe, and one must be very careful if the payload field of
  // empty locations are also used for other purposes (which is not the case so far).
  unsigned short next_id() const {
    ++unique_id;
    if (__builtin_expect(unique_id < (1U << 14), 1)) {
      return unique_id;
    }
    // Reset the payload field of all empty space to 0.
    for (size_t loc = 0; loc < N * N; ++loc) {
      if (!board[loc].has_stone) {
        board[loc].payload = 0;
      }
    }
    return unique_id = 1;
  }

  std::array<Point, N * N> board{};
  mutable unsigned short unique_id = 0;
  ZobristHashType hash = 0;
  std::unordered_set<ZobristHash::type> seen_states;
  const float komi;
  const std::unordered_set<ZobristHash::type>* existing_states;
};
}  // namespace go_engine
#endif  // #ifndef INCLUDE_GUARD_BOARD_H__
