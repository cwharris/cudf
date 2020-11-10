#include <cstdint>
#include <type_traits>

namespace cudf {
namespace io {
namespace detail {

// ===== STATES AND TOKENS =====

enum class csv_token : uint8_t {
  other,
  delimiter,
  quote,
  newline,
  comment,
};

enum class csv_state : uint8_t {
  record_end,
  comment,
  field,
  field_end,
  field_quoted,
  field_quoted_quote,
  none,
};

// ===== TRANSITIONS =====

namespace {

template <csv_state, csv_token>
struct transition {
  static constexpr csv_state destination = csv_state::none;
};

#ifndef TRANSITION_DEFAULT
#define TRANSITION_DEFAULT(from, to)             \
  template <csv_token via>                       \
  struct transition<csv_state::from, via> {      \
    static constexpr csv_state destination = to; \
  }
#endif

#ifndef TRANSITION
#define TRANSITION(from, via, to)                      \
  template <>                                          \
  struct transition<csv_state::from, csv_token::via> { \
    static constexpr csv_state destination = to;       \
  }
#endif

TRANSITION_DEFAULT(comment, csv_state::comment);
TRANSITION(comment, newline, csv_state::record_end);
TRANSITION(record_end, newline, csv_state::record_end);
TRANSITION(record_end, comment, csv_state::comment);
TRANSITION(record_end, other, csv_state::field);
TRANSITION(record_end, quote, csv_state::field_quoted);
TRANSITION(field, other, csv_state::field);
TRANSITION(field, delimiter, csv_state::field_end);
TRANSITION(field, newline, csv_state::record_end);
TRANSITION(field_end, other, csv_state::field);
TRANSITION(field_end, newline, csv_state::record_end);
TRANSITION(field_end, quote, csv_state::field_quoted);
TRANSITION(field_quoted, quote, csv_state::field_quoted_quote);
TRANSITION(field_quoted, other, csv_state::field_quoted);
TRANSITION(field_quoted, newline, csv_state::field_quoted);
TRANSITION(field_quoted, delimiter, csv_state::field_quoted);
TRANSITION(field_quoted, comment, csv_state::field_quoted);
TRANSITION(field_quoted_quote, quote, csv_state::field_quoted);
TRANSITION(field_quoted_quote, delimiter, csv_state::field_end);
TRANSITION(field_quoted_quote, newline, csv_state::record_end);

// ===== DISPATCH =====

template <csv_state state, csv_token token>
inline constexpr csv_state get_next_state()
{
  return transition<state, token>::destination;
}

template <csv_state state>
inline constexpr csv_state get_next_state(csv_token token)
{
  // compiler can optimize switch statements in to a jump list if cases are continuous.
  // keep these in the order in which they are declared.
  switch (token) {
    case (csv_token::other):  //
      return get_next_state<state, csv_token::other>();
    case (csv_token::delimiter):  //
      return get_next_state<state, csv_token::delimiter>();
    case (csv_token::quote):  //
      return get_next_state<state, csv_token::quote>();
    case (csv_token::newline):  //
      return get_next_state<state, csv_token::newline>();
    case (csv_token::comment):  //
      return get_next_state<state, csv_token::comment>();
  }

  return csv_state::none;
}

}  // namespace

inline constexpr csv_state get_next_state(csv_state state, csv_token token)
{
  // compiler can optimize switch statements in to a jump list if cases are continuous.
  // keep these in the order in which they are declared.
  switch (state) {
    case (csv_state::record_end):  //
      return get_next_state<csv_state::record_end>(token);
    case (csv_state::comment):  //
      return get_next_state<csv_state::comment>(token);
    case (csv_state::field):  //
      return get_next_state<csv_state::field>(token);
    case (csv_state::field_end):  //
      return get_next_state<csv_state::field_end>(token);
    case (csv_state::field_quoted):  //
      return get_next_state<csv_state::field_quoted>(token);
    case (csv_state::field_quoted_quote):  //
      return get_next_state<csv_state::field_quoted_quote>(token);
    case (csv_state::none):  //
      return csv_state::none;
  }

  return csv_state::none;
}

inline constexpr csv_state operator+(csv_state state, csv_token token)
{
  return get_next_state(state, token);
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
