// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/pc.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using IStore = VStore<Itv, standard_allocator>;
using IPC = PC<IStore>; // Interval Propagators Completion

const AType sty = 0;
const AType pty = 1;

TEST(TermTest, AddTermBinary) {
  IStore store = interpret_tell_to2<IStore>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);");
  pc::Add<pc::Variable<IStore>, pc::Variable<IStore>> x_plus_y(
    pc::Variable<IStore>(AVar(sty, 0)),
    pc::Variable<IStore>(AVar(sty, 1)));
  EXPECT_EQ(x_plus_y.project(store), Itv(0,20));
  local::BInc has_changed2;
  x_plus_y.tell(store, Itv(zi::bot(), 5), has_changed2);
  EXPECT_TRUE(has_changed2);
  EXPECT_EQ(x_plus_y.project(store), Itv(0,10));
}

TEST(TermTest, AddTermNary) {
  IStore store = interpret_tell_to2<IStore>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_ge(z, 0); constraint int_le(z, 10);");
  vector<pc::Variable<IStore>, standard_allocator> vars(3);
  for(int i = 0; i < 3; ++i) {
    vars[i] = pc::Variable<IStore>(AVar(sty, i));
  }
  pc::NaryAdd<pc::Variable<IStore>, standard_allocator> sum_xyz(std::move(vars));
  EXPECT_EQ(sum_xyz.project(store), Itv(0,30));
  local::BInc has_changed2;
  sum_xyz.tell(store, Itv(zi::bot(), 5), has_changed2);
  EXPECT_TRUE(has_changed2);
  EXPECT_EQ(sum_xyz.project(store), Itv(0,15));
}

template <class L>
void test_extract(const L& ipc, bool is_ua) {
  AbstractDeps<standard_allocator> deps;
  L copy1(ipc, deps);
  EXPECT_EQ(ipc.extract(copy1), is_ua);
  EXPECT_EQ(ipc.is_top(), copy1.is_top());
  EXPECT_EQ(ipc.is_bot(), copy1.is_bot());
  for(int i = 0; i < ipc.vars(); ++i) {
    EXPECT_EQ(ipc[i], copy1[i]);
  }
}

template<class L>
void refine_and_test(L& ipc, int num_refine, const std::vector<Itv>& before, const std::vector<Itv>& after, bool is_ua, bool expect_changed = true) {
  EXPECT_EQ(ipc.num_refinements(), num_refine);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(ipc[i], before[i]);
  }
  local::BInc has_changed = GaussSeidelIteration{}.fixpoint(ipc);
  EXPECT_EQ(has_changed, expect_changed);
  for(int i = 0; i < after.size(); ++i) {
    EXPECT_EQ(ipc[i], after[i]);
  }
  test_extract(ipc, is_ua);
}

template<class L>
void refine_and_test(L& ipc, int num_refine, const std::vector<Itv>& before_after, bool is_ua = false) {
  refine_and_test(ipc, num_refine, before_after, before_after, is_ua, false);
}

// x + y = 5
TEST(IPCTest, AddEquality) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_plus(x, y, 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(0,5)}, false);
}

// x + y = z, z <= 5
TEST(IPCTest, TemporalConstraint1Flat) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(z, 5);\
    constraint int_plus(x, y, z);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10), Itv(zi::bot(), zd(5))}, {Itv(0,5), Itv(0,5), Itv(0,5)}, false);
}

TEST(IPCTest, TemporalConstraint1) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(0,5)}, false);
}

// x + y > 5 (x,y in [0..10])
TEST(IPCTest, TemporalConstraint2) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_gt(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, false);
}

// x + y > 5 (x,y in [0..3])
TEST(IPCTest, TemporalConstraint3) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 3);\
    constraint int_ge(y, 0); constraint int_le(y, 3);\
    constraint int_gt(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,3), Itv(0,3)}, {Itv(3,3), Itv(3,3)}, true);
}

// x + y >= 5 (x,y in [0..3])
TEST(IPCTest, TemporalConstraint4) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 3);\
    constraint int_ge(y, 0); constraint int_le(y, 3);\
    constraint int_ge(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,3), Itv(0,3)}, {Itv(2,3), Itv(2,3)}, false);
}

// x + y = 5 (x,y in [0..4])
TEST(IPCTest, TemporalConstraint5) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 4);\
    constraint int_ge(y, 0); constraint int_le(y, 4);\
    constraint int_eq(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,4), Itv(0,4)}, {Itv(1,4), Itv(1,4)}, false);
}

// x - y <= 5 (x,y in [0..10])
TEST(IPCTest, TemporalConstraint6) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(int_minus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, false);
}

// x - y <= -10 (x,y in [0..10])
TEST(IPCTest, TemporalConstraint7) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(int_minus(x, y), -10);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,0), Itv(10,10)}, true);
}

// x - y >= 5 (x,y in [0..10])
TEST(IPCTest, TemporalConstraint8) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_ge(int_minus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(5,10), Itv(0,5)}, false);
}

// x - y <= -5 (x,y in [0..10])
TEST(IPCTest, TemporalConstraint9) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(int_minus(x, y), -5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(5,10)}, false);
}

// x <= -5 + y (x,y in [0..10])
TEST(IPCTest, TemporalConstraint10) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(x, int_plus(-5, y));");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(5,10)}, false);
}

// TOP test x,y,z in [3..10] /\ x + y + z <= 8
TEST(IPCTest, TopProp) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 3); constraint int_le(x, 10);\
    constraint int_ge(y, 3); constraint int_le(y, 10);\
    constraint int_ge(z, 3); constraint int_le(z, 10);\
    constraint int_le(int_plus(int_plus(x, y),z), 8);");
  refine_and_test(ipc, 1, {Itv(3,10), Itv(3,10), Itv(3,10)}, {Itv::top(), Itv::top(), Itv::top()}, false);
  EXPECT_TRUE(ipc.is_top());
}

// x,y,z in [3..10] /\ x + y + z <= 9
TEST(IPCTest, TernaryAdd2) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 3); constraint int_le(x, 10);\
    constraint int_ge(y, 3); constraint int_le(y, 10);\
    constraint int_ge(z, 3); constraint int_le(z, 10);\
    constraint int_le(int_plus(int_plus(x, y),z), 9);");
  refine_and_test(ipc, 1, {Itv(3,10), Itv(3,10), Itv(3,10)}, {Itv(3,3), Itv(3,3), Itv(3,3)}, true);
}

// x,y,z in [3..10] /\ x + y + z <= 10
TEST(IPCTest, TernaryAdd3) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 3); constraint int_le(x, 10);\
    constraint int_ge(y, 3); constraint int_le(y, 10);\
    constraint int_ge(z, 3); constraint int_le(z, 10);\
    constraint int_le(int_plus(int_plus(x, y),z), 10);");
  refine_and_test(ipc, 1, {Itv(3,10), Itv(3,10), Itv(3,10)}, {Itv(3,4), Itv(3,4), Itv(3,4)}, false);
}

// x,y,z in [-2..2] /\ x + y + z <= -5
TEST(IPCTest, TernaryAdd4) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, -2); constraint int_le(x, 2);\
    constraint int_ge(y, -2); constraint int_le(y, 2);\
    constraint int_ge(z, -2); constraint int_le(z, 2);\
    constraint int_le(int_plus(int_plus(x, y),z), -5);");
  refine_and_test(ipc, 1, {Itv(-2,2), Itv(-2,2), Itv(-2,2)}, {Itv(-2,-1), Itv(-2,-1), Itv(-2,-1)}, false);
}

// x,y,z in [0..1] /\ 2x + y + 3z <= 2
TEST(IPCTest, PseudoBoolean1) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 1);\
    constraint int_ge(y, 0); constraint int_le(y, 1);\
    constraint int_ge(z, 0); constraint int_le(z, 1);\
    constraint int_le(int_plus(int_plus(int_times(2,x), y),int_times(3,z)), 2);");
  refine_and_test(ipc, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, {Itv(0,1), Itv(0,1), Itv(0,0)}, false);
}

// x,y,z in [0..1] /\ 2x + 5y + 3z <= 2
TEST(IPCTest, PseudoBoolean2) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 1);\
    constraint int_ge(y, 0); constraint int_le(y, 1);\
    constraint int_ge(z, 0); constraint int_le(z, 1);\
    constraint int_le(int_plus(int_plus(int_times(2,x), int_times(5,y)),int_times(3,z)), 2);");
  refine_and_test(ipc, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, {Itv(0,1), Itv(0,0), Itv(0,0)}, true);
}

// x,y,z in [0..1] /\ 3x + 5y + 3z <= 2
TEST(IPCTest, PseudoBoolean3) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 1);\
    constraint int_ge(y, 0); constraint int_le(y, 1);\
    constraint int_ge(z, 0); constraint int_le(z, 1);\
    constraint int_le(int_plus(int_plus(int_times(3,x), int_times(5,y)),int_times(3,z)), 2);");
  refine_and_test(ipc, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, {Itv(0,0), Itv(0,0), Itv(0,0)}, true);
}

// x,y,z in [0..1] /\ -x + y + 3z <= 2
TEST(IPCTest, PseudoBoolean4) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 1);\
    constraint int_ge(y, 0); constraint int_le(y, 1);\
    constraint int_ge(z, 0); constraint int_le(z, 1);\
    constraint int_le(int_plus(int_plus(int_neg(x), y),int_times(3,z)), 2);");
  refine_and_test(ipc, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, false);
}

// x in [-4..3], -x <= 2
TEST(IPCTest, NegationOp1) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, -4); constraint int_le(x, 3);\
    constraint int_le(int_neg(x), 2);");
  refine_and_test(ipc, 1, {Itv(-4,3)}, {Itv(-2,3)}, true);
}

// x in [-4..3], -x <= -2
TEST(IPCTest, NegationOp2) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, -4); constraint int_le(x, 3);\
    constraint int_le(int_neg(x), -2);");
  refine_and_test(ipc, 1, {Itv(-4,3)}, {Itv(2,3)}, true);
}

// x in [0..3], -x <= -2
TEST(IPCTest, NegationOp3) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, 0); constraint int_le(x, 3);\
    constraint int_le(int_neg(x), -2);");
  refine_and_test(ipc, 1, {Itv(0,3)}, {Itv(2,3)}, true);
}

// x in [-4..-3], -x <= 4
TEST(IPCTest, NegationOp4) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, -4); constraint int_le(x, -3);\
    constraint int_le(int_neg(x), 4);");
  refine_and_test(ipc, 1, {Itv(-4,-3)}, true);
}

// x in [-4..3], -x >= -2
TEST(IPCTest, NegationOp5) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, -4); constraint int_le(x, 3);\
    constraint int_ge(int_neg(x), -2);");
  refine_and_test(ipc, 1, {Itv(-4,3)}, {Itv(-4,2)}, true);
}

// x in [-4..3], -x > 2
TEST(IPCTest, NegationOp6) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, -4); constraint int_le(x, 3);\
    constraint int_gt(int_neg(x), 2);");
  refine_and_test(ipc, 1, {Itv(-4,3)}, {Itv(-4,-3)}, true);
}

// x in [-4..3], -x >= 5
TEST(IPCTest, NegationOp7) {
  IPC ipc = interpret_tell_to2<IPC>("var int: x;\
    constraint int_ge(x, -4); constraint int_le(x, 3);\
    constraint int_ge(int_neg(x), 5);");
  refine_and_test(ipc, 1, {Itv(-4,3)}, {Itv::top()}, false);
  EXPECT_TRUE(ipc.is_top());
}

// Constraint of the form "b <=> (x - y <= k1 /\ y - x <= k2)".
TEST(IPCTest, ResourceConstraint1) {
  VarEnv<standard_allocator> env;
  IPC ipc = interpret_tell_to<IPC>("var int: x; var int: y; var int: b;\
    constraint int_ge(x, 5); constraint int_le(x, 10);\
    constraint int_ge(y, 9); constraint int_le(y, 15);\
    constraint int_ge(b, 0); constraint int_le(b, 1);\
    constraint int_eq(b, bool_and(int_le(int_minus(x, y), 0), int_le(int_minus(y, x), 2)));", env);
  refine_and_test(ipc, 1, {Itv(5,10), Itv(9,15), Itv(0,1)}, false);

  interpret_and_tell(ipc, "constraint int_eq(b, 1);", env);
  refine_and_test(ipc, 1, {Itv(5,10), Itv(9,15), Itv(1,1)}, {Itv(7,10), Itv(9,12), Itv(1,1)}, false);
}

TEST(IPCTest, ResourceConstraint2) {
  VarEnv<standard_allocator> env;
  IPC ipc = interpret_tell_to<IPC>("var int: x; var int: y; var int: b;\
    constraint int_ge(x, 1); constraint int_le(x, 2);\
    constraint int_ge(y, 0); constraint int_le(y, 2);\
    constraint int_ge(b, 0); constraint int_le(b, 1);\
    constraint int_eq(b, bool_and(int_le(int_minus(x, y), 2), int_le(int_minus(y, x), -1)));", env);
  refine_and_test(ipc, 1, {Itv(1,2), Itv(0,2), Itv(0,1)}, false);

  interpret_and_tell(ipc, "constraint int_eq(b, 0);", env);
  refine_and_test(ipc, 1, {Itv(1,2), Itv(0,2), Itv(0,0)}, {Itv(1,2), Itv(1,2), Itv(0,0)}, false);
}

TEST(IPCTest, NotEqualConstraint1) {
  VarEnv<standard_allocator> env;
  IPC ipc = interpret_tell_to<IPC>("var 1..10: x; constraint int_ne(x, 10);", env);
  refine_and_test(ipc, 1, {Itv(1,10)}, {Itv(1,9)}, true);
}

TEST(IPCTest, NotEqualConstraint2) {
  VarEnv<standard_allocator> env;
  IPC ipc = interpret_tell_to<IPC>("var 1..10: x; var 10..10: y; constraint int_ne(x, y);", env);
  refine_and_test(ipc, 1, {Itv(1,10), Itv(10,10)}, {Itv(1,9), Itv(10,10)}, true);
}