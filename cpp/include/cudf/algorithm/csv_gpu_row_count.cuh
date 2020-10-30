// #pragma once

// #include <cstdint>
// #include <type_traits>

// #include <cudf/algorithm/scan_artifacts.cuh>
// #include <cudf/types.hpp>
// #include <cudf/utilities/span.hpp>

// #include <rmm/device_uvector.hpp>
// #include <rmm/mr/device/per_device_resource.hpp>

// using cudf::detail::device_span;

// // ===== STATES AND TOKENS =====

// enum class csv_token : uint8_t {
//   comment         = 0b00000001,
//   delimiter       = 0b00000010,
//   whitespace      = 0b00000100,
//   quote           = 0b00001000,
//   newline         = 0b00010000,
//   other           = 0b00100000,
//   carriage_return = 0b10000000,
// };

// enum class csv_state : uint16_t {
//   none               = 0b0000000000000000,
//   record_end         = 0b0000000000000001,
//   comment            = 0b0000000000000010,
//   whitespace_pre     = 0b0000000000000100,
//   whitespace_post    = 0b0000000000001000,
//   field              = 0b0000000000010000,
//   field_quoted       = 0b0000000000100000,
//   field_quoted_quote = 0b0000000010000000,
//   field_end          = 0b0000000100000000,
//   end                = 0b0000001000000000,
//   carriage_end       = 0b0000100000000000,
//   all                = 0b1111111111111111
// };

// constexpr csv_state csv_state_loop_values[] = {
//   csv_state::none,
//   csv_state::record_end,
//   csv_state::comment,
//   csv_state::whitespace_pre,
//   csv_state::whitespace_post,
//   csv_state::field,
//   csv_state::field_quoted,
//   csv_state::field_quoted_quote,
//   csv_state::field_end,
//   csv_state::end,
//   csv_state::carriage_end,
// };
// constexpr csv_state operator|(csv_state lhs, csv_state rhs)
// {
//   using underlying = std::underlying_type_t<csv_state>;
//   return static_cast<csv_state>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
// }

// constexpr csv_state operator&(csv_state lhs, csv_state rhs)
// {
//   using underlying = std::underlying_type_t<csv_state>;
//   return static_cast<csv_state>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
// }

// // ===== TRANSITIONS =====

// template <csv_state, csv_token>
// struct transition {
//   static constexpr csv_state destination = csv_state::none;
// };
/*
// #ifndef TRANSITION_DEFAULT
// #define TRANSITION_DEFAULT(from, to)             \
//   template <csv_token via>                       \
//   struct transition<csv_state::from, via> {      \
//     static constexpr csv_state destination = to; \
//   }
// #endif

// #ifndef TRANSITION
// #define TRANSITION(from, via, to)                      \
//   template <>                                          \
//   struct transition<csv_state::from, csv_token::via> { \
//     static constexpr csv_state destination = to;       \
//   }
// #endif
*/
// TRANSITION_DEFAULT(comment, csv_state::comment);
// TRANSITION(comment, newline, csv_state::record_end);
// TRANSITION(record_end, newline, csv_state::record_end);
// TRANSITION(record_end, carriage_return, csv_state::record_end);
// TRANSITION(record_end, comment, csv_state::comment);
// TRANSITION(record_end, other, csv_state::field);
// TRANSITION(record_end, whitespace, csv_state::whitespace_pre);
// TRANSITION(field, other, csv_state::field);
// TRANSITION(field, whitespace, csv_state::whitespace_post | csv_state::field);
// TRANSITION(field, delimiter, csv_state::field_end);
// TRANSITION(field_end, other, csv_state::field);
// TRANSITION(field_end, whitespace, csv_state::whitespace_pre | csv_state::field);
// TRANSITION(field_end, newline, csv_state::record_end);
// TRANSITION(field_end, carriage_return, csv_state::carriage_end);
// TRANSITION(carriage_end, newline, csv_state::record_end);

// // ===== DISPATCH =====

// template <csv_state state, csv_token token>
// constexpr csv_state dispatch()
// {
//   return transition<state, token>::destination;
// }

// template <csv_state state>
// constexpr csv_state dispatch(csv_token token)
// {
//   switch (token) {
//     case (csv_token::comment):  //
//       return dispatch<state, csv_token::comment>();
//     case (csv_token::whitespace):  //
//       return dispatch<state, csv_token::whitespace>();
//     case (csv_token::other):  //
//       return dispatch<state, csv_token::other>();
//     case (csv_token::newline):  //
//       return dispatch<state, csv_token::newline>();
//     case (csv_token::delimiter):  //
//       return dispatch<state, csv_token::delimiter>();
//     case (csv_token::quote):  //
//       return dispatch<state, csv_token::quote>();
//     case (csv_token::carriage_return):  //
//       return dispatch<state, csv_token::carriage_return>();
//   }

//   return csv_state::none;
// }

// constexpr csv_state dispatch(csv_state state, csv_token token)
// {
//   switch (state) {
//     case (csv_state::field_quoted_quote):  //
//       return dispatch<csv_state::field_quoted_quote>(token);
//     case (csv_state::field_quoted):  //
//       return dispatch<csv_state::field_quoted>(token);
//     case (csv_state::whitespace_pre):  //
//       return dispatch<csv_state::whitespace_pre>(token);
//     case (csv_state::whitespace_post):  //
//       return dispatch<csv_state::whitespace_post>(token);
//     case (csv_state::field_end):  //
//       return dispatch<csv_state::field_end>(token);
//     case (csv_state::record_end):  //
//       return dispatch<csv_state::record_end>(token);
//     case (csv_state::comment):  //
//       return dispatch<csv_state::comment>(token);
//     case (csv_state::field):  //
//       return dispatch<csv_state::field>(token);
//     case (csv_state::none):  //
//       return csv_state::none;
//     case (csv_state::end):  //
//       return csv_state::none;
//     case (csv_state::carriage_end):  //
//       return dispatch<csv_state::carriage_end>(token);
//   }

//   return csv_state::none;
// }

// // ===== PARSING =====

// constexpr csv_token get_token(char prev, char current)
// {
//   if (prev != '\\') {
//     // what happens if we see `\` twice? i.e: "\\\\" <- this should be two backslashes
//     // maybe take care of this elsewhere.
//     switch (current) {
//       case '\n': return csv_token::newline;
//       case '\r': return csv_token::carriage_return;
//       case '#': return csv_token::comment;
//       case ',': return csv_token::delimiter;
//       case '"': return csv_token::quote;
//       case ' ': return csv_token::whitespace;
//     }
//   }

//   return csv_token::other;
// }

// constexpr csv_state get_next(csv_state current, csv_token token)
// {
//   csv_state result = dispatch(current, token);

//   for (auto state : csv_state_loop_values) {
//     if ((current & state) == state) { result = result | dispatch(state, token); }
//   }

//   return result;
// }

// // ===== parallel state machine for CSV parsing =====W

// struct csv_machine_state {
//   uint64_t idx;
//   csv_state current_state;
// };

// struct csv_row_count_seed_op {
//   csv_machine_state operator()(uint32_t idx, char current_char)  //
//   {
//     // If this is the start of the csv, then use the starting state.
//     // Otherwise, use all possible states, and eliminate erroneous states during scan +
//     // intersection. You know, like quantum physics. ;)
//     auto state = idx == 0 ? csv_state::carriage_end : csv_state::all;
//     // once we know the possible prior states, we can transition in to the current one.
//     auto token = get_token(0, current_char);
//     return {idx, get_next(state, token)};
//   }
// };

// struct csv_row_count_scan_op {
//   template <typename Callback>
//   csv_machine_state operator()(csv_machine_state state, char current_char, Callback output)  //
//   {
//     if (state.current_state == csv_state::record_end) {
//       output(state.idx);  // output the record offsets!
//     }

//     auto token = get_token(0, current_char);

//     return {state.idx + 1, get_next(state.current_state, token)};
//   }
// };

// struct csv_row_count_intersection_op {
//   csv_machine_state operator()(csv_machine_state lhs, csv_machine_state rhs)  //
//   {
//     // the rhs states must be a subset of the lhs states.
//     // we use binary-and to eliminate erroneous states.
//     // the result of this function is the "real" rhs state.
//     return {
//       rhs.idx,
//       lhs.current_state & rhs.current_state,
//     };
//   }
// };

// // rmm::device_uvector<uint32_t> csv_gather_row_offsets(
// //   device_span<char> csv_input,
// //   rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
// // {
// //   auto seed_op         = csv_row_count_seed_op{};
// //   auto scan_op         = csv_row_count_scan_op{};
// //   auto intersection_op = csv_row_count_intersection_op{};

// //   return scan_artifacts<uint32_t>(csv_input.begin(),  //
// //                                   csv_input.end(),
// //                                   seed_op,
// //                                   scan_op,
// //                                   intersection_op);
// // }
