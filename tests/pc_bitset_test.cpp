// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/universes/nbitset.hpp"
#include "lala/pc.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

using NBit = NBitset<64, battery::local_memory, unsigned long long>;
using BitStore = VStore<NBit, standard_allocator>;
using BitPC = PC<BitStore>; // NBitset Propagators Completion

const AType sty = 0;
const AType pty = 1;

template <class L>
void test_extract(const L& bpc, bool is_ua) {
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  L copy1(bpc, deps);
  EXPECT_EQ(bpc.is_extractable(), is_ua);
  if(bpc.is_extractable()) {
    bpc.extract(copy1);
    EXPECT_EQ(bpc.is_top(), copy1.is_top());
    EXPECT_EQ(bpc.is_bot(), copy1.is_bot());
    for(int i = 0; i < bpc.vars(); ++i) {
      EXPECT_EQ(bpc[i], copy1[i]);
    }
  }
}

template<class L>
void deduce_and_test(L& bpc, int num_deds, const std::vector<NBit>& before, const std::vector<NBit>& after, bool is_ua, bool expect_changed = true) {
  EXPECT_EQ(bpc.num_deductions(), num_deds);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(bpc[i], before[i]) << "bpc[" << i << "]";
  }
  local::B has_changed = false;
  GaussSeidelIteration{}.fixpoint(
    bpc.num_deductions(),
    [&](size_t i) { return bpc.deduce(i); },
    has_changed);
  EXPECT_EQ(has_changed, expect_changed);
  for(int i = 0; i < after.size(); ++i) {
    EXPECT_EQ(bpc[i], after[i]) << "bpc[" << i << "]";
  }
  test_extract(bpc, is_ua);
}

template<class L>
void deduce_and_test(L& bpc, int num_deds, const std::vector<NBit>& before_after, bool is_ua = false) {
  deduce_and_test(bpc, num_deds, before_after, before_after, is_ua, false);
}

TEST(BitPCTest, NotEqualConstraint1) {
  VarEnv<standard_allocator> env;
  BitPC bpc = create_and_interpret_and_tell<BitPC>("var 1..10: x; constraint int_ne(x, 10);", env);
  deduce_and_test(bpc, 0, {NBit(1,9)}, {NBit(1,9)}, true, false);
}

TEST(BitPCTest, NotEqualConstraint2) {
  VarEnv<standard_allocator> env;
  BitPC bpc = create_and_interpret_and_tell<BitPC>("var 1..10: x; var 10..10: y; constraint int_ne(x, y);", env);
  deduce_and_test(bpc, 1, {NBit(1,10), NBit(10,10)}, {NBit(1,9), NBit(10,10)}, true);
}

TEST(BitPCTest, NotEqualConstraint3) {
  VarEnv<standard_allocator> env;
  BitPC bpc = create_and_interpret_and_tell<BitPC>("var 1..10: x; constraint int_ne(x, 4);", env);
  deduce_and_test(bpc, 0, {NBit::from_set({1,2,3,5,6,7,8,9,10})}, {NBit::from_set({1,2,3,5,6,7,8,9,10})}, true, false);
}

TEST(BitPCTest, NotEqualConstraint4) {
  VarEnv<standard_allocator> env;
  BitPC bpc = create_and_interpret_and_tell<BitPC>("var 1..10: x; var 4..4: y; constraint int_ne(x, y);", env);
  deduce_and_test(bpc, 1, {NBit(1,10), NBit(4,4)}, {NBit::from_set({1,2,3,5,6,7,8,9,10}), NBit(4)}, true);
}

// Constraint of the form "x in {1,3}".
TEST(BitPCTest, InConstraint1) {
  VarEnv<standard_allocator> env;
  BitPC bpc = create_and_interpret_and_tell<BitPC>("var {1, 3}: x; var 2..3: y;", env);
  deduce_and_test(bpc, 0, {NBit::from_set({1, 3}), NBit(2,3)}, true);

  interpret_must_succeed<IKind::TELL>("constraint int_eq(x, y);", bpc, env);
  deduce_and_test(bpc, 1, {NBit::from_set({1, 3}), NBit(2,3)}, {NBit(3), NBit(3)}, true);
}

TEST(BitPCTest, BooleanClause1) {
  VarEnv<standard_allocator> env;
  BitPC bpc =  create_and_interpret_and_tell<BitPC>("array[1..2] of var bool: x;\
    array[1..2] of var bool: y;\
    constraint bool_clause(x, y);", env);
  deduce_and_test(bpc, 1, {NBit(0, 1), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(x[1], true);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(1, 1), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, true);
}

TEST(BitPCTest, BooleanClause2) {
  VarEnv<standard_allocator> env;
  BitPC bpc =  create_and_interpret_and_tell<BitPC>("array[1..2] of var bool: x;\
    array[1..2] of var bool: y;\
    constraint bool_clause(x, y);", env);
  deduce_and_test(bpc, 1, {NBit(0, 1), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(y[1], false);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 1), NBit(0, 1), NBit(0, 0), NBit(0, 1)}, true);
}

TEST(BitPCTest, BooleanClause3) {
  VarEnv<standard_allocator> env;
  BitPC bpc =  create_and_interpret_and_tell<BitPC>("array[1..2] of var bool: x;\
    array[1..2] of var bool: y;\
    constraint bool_clause(x, y);", env);
  deduce_and_test(bpc, 1, {NBit(0, 1), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(x[1], false);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 0), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(x[2], false);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 0), NBit(0, 0), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(y[1], true);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 0), NBit(0, 0), NBit(1, 1), NBit(0, 1)},  {NBit(0, 0), NBit(0, 0), NBit(1, 1), NBit(0, 0)}, true);
}

TEST(BitPCTest, BooleanClause4) {
  VarEnv<standard_allocator> env;
  BitPC bpc =  create_and_interpret_and_tell<BitPC>("array[1..2] of var bool: x;\
    array[1..2] of var bool: y;\
    constraint bool_clause(x, y);", env);
  deduce_and_test(bpc, 1, {NBit(0, 1), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(x[1], false);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 0), NBit(0, 1), NBit(0, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(y[1], true);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 0), NBit(0, 1), NBit(1, 1), NBit(0, 1)}, false);
  interpret_must_succeed<IKind::TELL>("constraint int_eq(y[2], true);", bpc, env);
  deduce_and_test(bpc, 1, {NBit(0, 0), NBit(0, 1), NBit(1, 1), NBit(1, 1)},  {NBit(0, 0), NBit(1, 1), NBit(1, 1), NBit(1, 1)}, true);
}

TEST(BitPCTest, IntAbs1) {
  VarEnv<standard_allocator> env;
  BitPC bpc =  create_and_interpret_and_tell<BitPC>("\
    var -15..5: x;\
    var -10..10: y;\
    constraint int_abs(x, y);", env, true);
  deduce_and_test(bpc, 1, {NBit(-1, 5), NBit(-1, 10)}, {NBit(-1, 5), NBit(0, 10)}, false);
}
