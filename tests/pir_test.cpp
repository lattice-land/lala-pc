// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/pir.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

using zlb = local::ZLB;
using zub = local::ZUB;
using Itv = Interval<zlb>;
using IStore = VStore<Itv, standard_allocator>;
using IPIR = PIR<IStore>; // Interval Propagators Completion

const AType sty = 0;
const AType pty = 1;

template <class L>
void test_extract(const L& pir, bool is_ua) {
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  L copy1(pir, deps);
  EXPECT_EQ(pir.is_extractable(), is_ua);
  if(pir.is_extractable()) {
    pir.extract(copy1);
    EXPECT_EQ(pir.is_top(), copy1.is_top());
    EXPECT_EQ(pir.is_bot(), copy1.is_bot());
    for(int i = 0; i < pir.vars(); ++i) {
      EXPECT_EQ(pir[i], copy1[i]);
    }
  }
}

template<class L>
void deduce_and_test(L& pir, int num_deds, const std::vector<Itv>& before, const std::vector<Itv>& after, bool is_ua, bool expect_changed = true) {
  EXPECT_EQ(pir.num_deductions(), num_deds);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(pir[i], before[i]) << "pir[" << i << "]";
  }
  local::B has_changed = GaussSeidelIteration{}.fixpoint(pir);
  EXPECT_EQ(has_changed, expect_changed);
  for(int i = 0; i < after.size(); ++i) {
    EXPECT_EQ(pir[i], after[i]) << "pir[" << i << "]";
  }
  test_extract(pir, is_ua);
}

template<class L>
void deduce_and_test(L& pir, int num_deds, const std::vector<Itv>& before_after, bool is_ua = false) {
  deduce_and_test(pir, num_deds, before_after, before_after, is_ua, false);
}

TEST(PIRTest, BinarizedProblem) {
  IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_ge(z, 5); constraint int_le(z, 5);\
    constraint int_plus(x, y, z);");
  deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10), Itv(5,5)}, {Itv(0,5), Itv(0,5), Itv(5,5)}, false);
}

// // x + y = 5
// TEST(PIRTest, AddEquality) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_plus(x, y, 5);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(0,5)}, false);
// }

// // x + y = z, z <= 5
// TEST(PIRTest, TemporalConstraint1Flat) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_le(z, 5);\
//     constraint int_plus(x, y, z);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10), Itv(zlb::top(), zub(5))}, {Itv(0,5), Itv(0,5), Itv(0,5)}, false);
// }

// TEST(PIRTest, TemporalConstraint1) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_le(int_plus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(0,5)}, false);
// }

// // x + y > 5 (x,y in [0..10])
// TEST(PIRTest, TemporalConstraint2) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_gt(int_plus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, false);
// }

// // x + y > 5 (x,y in [0..3])
// TEST(PIRTest, TemporalConstraint3) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 3);\
//     constraint int_ge(y, 0); constraint int_le(y, 3);\
//     constraint int_gt(int_plus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,3), Itv(0,3)}, {Itv(3,3), Itv(3,3)}, true);
// }

// // x + y >= 5 (x,y in [0..3])
// TEST(PIRTest, TemporalConstraint4) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 3);\
//     constraint int_ge(y, 0); constraint int_le(y, 3);\
//     constraint int_ge(int_plus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,3), Itv(0,3)}, {Itv(2,3), Itv(2,3)}, false);
// }

// // x + y = 5 (x,y in [0..4])
// TEST(PIRTest, TemporalConstraint5) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 4);\
//     constraint int_ge(y, 0); constraint int_le(y, 4);\
//     constraint int_eq(int_plus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,4), Itv(0,4)}, {Itv(1,4), Itv(1,4)}, false);
// }

// // x - y <= 5 (x,y in [0..10])
// TEST(PIRTest, TemporalConstraint6) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_le(int_minus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, false);
// }

// // x - y <= -10 (x,y in [0..10])
// TEST(PIRTest, TemporalConstraint7) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_le(int_minus(x, y), -10);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,0), Itv(10,10)}, true);
// }

// // x - y >= 5 (x,y in [0..10])
// TEST(PIRTest, TemporalConstraint8) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_ge(int_minus(x, y), 5);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, {Itv(5,10), Itv(0,5)}, false);
// }

// // x - y <= -5 (x,y in [0..10])
// TEST(PIRTest, TemporalConstraint9) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_le(int_minus(x, y), -5);");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(5,10)}, false);
// }

// // x <= -5 + y (x,y in [0..10])
// TEST(PIRTest, TemporalConstraint10) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint int_ge(x, 0); constraint int_le(x, 10);\
//     constraint int_ge(y, 0); constraint int_le(y, 10);\
//     constraint int_le(x, int_plus(-5, y));");
//   deduce_and_test(pir, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(5,10)}, false);
// }

// // TOP test x,y,z in [3..10] /\ x + y + z <= 8
// TEST(PIRTest, TopProp) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 3); constraint int_le(x, 10);\
//     constraint int_ge(y, 3); constraint int_le(y, 10);\
//     constraint int_ge(z, 3); constraint int_le(z, 10);\
//     constraint int_le(int_plus(int_plus(x, y),z), 8);");
//   deduce_and_test(pir, 1, {Itv(3,10), Itv(3,10), Itv(3,10)}, {Itv::bot(), Itv::bot(), Itv::bot()}, false);
//   EXPECT_TRUE(pir.is_bot());
// }

// // x,y,z in [3..10] /\ x + y + z <= 9
// TEST(PIRTest, TernaryAdd2) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 3); constraint int_le(x, 10);\
//     constraint int_ge(y, 3); constraint int_le(y, 10);\
//     constraint int_ge(z, 3); constraint int_le(z, 10);\
//     constraint int_le(int_plus(int_plus(x, y),z), 9);");
//   deduce_and_test(pir, 1, {Itv(3,10), Itv(3,10), Itv(3,10)}, {Itv(3,3), Itv(3,3), Itv(3,3)}, true);
// }

// // x,y,z in [3..10] /\ x + y + z <= 10
// TEST(PIRTest, TernaryAdd3) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 3); constraint int_le(x, 10);\
//     constraint int_ge(y, 3); constraint int_le(y, 10);\
//     constraint int_ge(z, 3); constraint int_le(z, 10);\
//     constraint int_le(int_plus(int_plus(x, y),z), 10);");
//   deduce_and_test(pir, 1, {Itv(3,10), Itv(3,10), Itv(3,10)}, {Itv(3,4), Itv(3,4), Itv(3,4)}, false);
// }

// // x,y,z in [-2..2] /\ x + y + z <= -5
// TEST(PIRTest, TernaryAdd4) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, -2); constraint int_le(x, 2);\
//     constraint int_ge(y, -2); constraint int_le(y, 2);\
//     constraint int_ge(z, -2); constraint int_le(z, 2);\
//     constraint int_le(int_plus(int_plus(x, y),z), -5);");
//   deduce_and_test(pir, 1, {Itv(-2,2), Itv(-2,2), Itv(-2,2)}, {Itv(-2,-1), Itv(-2,-1), Itv(-2,-1)}, false);
// }

// // x,y,z in [0..1] /\ 2x + y + 3z <= 2
// TEST(PIRTest, PseudoBoolean1) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 0); constraint int_le(x, 1);\
//     constraint int_ge(y, 0); constraint int_le(y, 1);\
//     constraint int_ge(z, 0); constraint int_le(z, 1);\
//     constraint int_le(int_plus(int_plus(int_times(2,x), y),int_times(3,z)), 2);");
//   deduce_and_test(pir, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, {Itv(0,1), Itv(0,1), Itv(0,0)}, false);
// }

// // x,y,z in [0..1] /\ 2x + 5y + 3z <= 2
// TEST(PIRTest, PseudoBoolean2) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 0); constraint int_le(x, 1);\
//     constraint int_ge(y, 0); constraint int_le(y, 1);\
//     constraint int_ge(z, 0); constraint int_le(z, 1);\
//     constraint int_le(int_plus(int_plus(int_times(2,x), int_times(5,y)),int_times(3,z)), 2);");
//   deduce_and_test(pir, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, {Itv(0,1), Itv(0,0), Itv(0,0)}, true);
// }

// // x,y,z in [0..1] /\ 3x + 5y + 3z <= 2
// TEST(PIRTest, PseudoBoolean3) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 0); constraint int_le(x, 1);\
//     constraint int_ge(y, 0); constraint int_le(y, 1);\
//     constraint int_ge(z, 0); constraint int_le(z, 1);\
//     constraint int_le(int_plus(int_plus(int_times(3,x), int_times(5,y)),int_times(3,z)), 2);");
//   deduce_and_test(pir, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, {Itv(0,0), Itv(0,0), Itv(0,0)}, true);
// }

// // x,y,z in [0..1] /\ -x + y + 3z <= 2
// TEST(PIRTest, PseudoBoolean4) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: z;\
//     constraint int_ge(x, 0); constraint int_le(x, 1);\
//     constraint int_ge(y, 0); constraint int_le(y, 1);\
//     constraint int_ge(z, 0); constraint int_le(z, 1);\
//     constraint int_le(int_plus(int_plus(int_neg(x), y),int_times(3,z)), 2);");
//   deduce_and_test(pir, 1, {Itv(0,1), Itv(0,1), Itv(0,1)}, false);
// }

// // x in [-4..3], -x <= 2
// TEST(PIRTest, NegationOp1) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, -4); constraint int_le(x, 3);\
//     constraint int_le(int_neg(x), 2);");
//   deduce_and_test(pir, 1, {Itv(-4,3)}, {Itv(-2,3)}, true);
// }

// // x in [-4..3], -x <= -2
// TEST(PIRTest, NegationOp2) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, -4); constraint int_le(x, 3);\
//     constraint int_le(int_neg(x), -2);");
//   deduce_and_test(pir, 1, {Itv(-4,3)}, {Itv(2,3)}, true);
// }

// // x in [0..3], -x <= -2
// TEST(PIRTest, NegationOp3) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, 0); constraint int_le(x, 3);\
//     constraint int_le(int_neg(x), -2);");
//   deduce_and_test(pir, 1, {Itv(0,3)}, {Itv(2,3)}, true);
// }

// // x in [-4..-3], -x <= 4
// TEST(PIRTest, NegationOp4) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, -4); constraint int_le(x, -3);\
//     constraint int_le(int_neg(x), 4);");
//   deduce_and_test(pir, 1, {Itv(-4,-3)}, true);
// }

// // x in [-4..3], -x >= -2
// TEST(PIRTest, NegationOp5) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, -4); constraint int_le(x, 3);\
//     constraint int_ge(int_neg(x), -2);");
//   deduce_and_test(pir, 1, {Itv(-4,3)}, {Itv(-4,2)}, true);
// }

// // x in [-4..3], -x > 2
// TEST(PIRTest, NegationOp6) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, -4); constraint int_le(x, 3);\
//     constraint int_gt(int_neg(x), 2);");
//   deduce_and_test(pir, 1, {Itv(-4,3)}, {Itv(-4,-3)}, true);
// }

// // x in [-4..3], -x >= 5
// TEST(PIRTest, NegationOp7) {
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x;\
//     constraint int_ge(x, -4); constraint int_le(x, 3);\
//     constraint int_ge(int_neg(x), 5);");
//   deduce_and_test(pir, 1, {Itv(-4,3)}, {Itv::bot()}, false);
//   EXPECT_TRUE(pir.is_bot());
// }

// // Constraint of the form "b <=> (x - y <= k1 /\ y - x <= k2)".
// TEST(PIRTest, ResourceConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: b;\
//     constraint int_ge(x, 5); constraint int_le(x, 10);\
//     constraint int_ge(y, 9); constraint int_le(y, 15);\
//     constraint int_ge(b, 0); constraint int_le(b, 1);\
//     constraint int_eq(b, bool_and(int_le(int_minus(x, y), 0), int_le(int_minus(y, x), 2)));", env);
//   deduce_and_test(pir, 1, {Itv(5,10), Itv(9,15), Itv(0,1)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(5,10), Itv(9,15), Itv(1,1)}, {Itv(7,10), Itv(9,12), Itv(1,1)}, false);
// }

// TEST(PIRTest, ResourceConstraint2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y; var int: b;\
//     constraint int_ge(x, 1); constraint int_le(x, 2);\
//     constraint int_ge(y, 0); constraint int_le(y, 2);\
//     constraint int_ge(b, 0); constraint int_le(b, 1);\
//     constraint int_eq(b, bool_and(int_le(int_minus(x, y), 2), int_le(int_minus(y, x), -1)));", env);
//   deduce_and_test(pir, 1, {Itv(1,2), Itv(0,2), Itv(0,1)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b, 0);", pir, env);
//   deduce_and_test(pir, 1, {Itv(1,2), Itv(0,2), Itv(0,0)}, {Itv(1,2), Itv(1,2), Itv(0,0)}, false);
// }

// TEST(PIRTest, NotEqualConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var 1..10: x; constraint int_ne(x, 10);", env);
//   deduce_and_test(pir, 1, {Itv(1,10)}, {Itv(1,9)}, true);
// }

// TEST(PIRTest, NotEqualConstraint2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var 1..10: x; var 10..10: y; constraint int_ne(x, y);", env);
//   deduce_and_test(pir, 1, {Itv(1,10), Itv(10,10)}, {Itv(1,9), Itv(10,10)}, true);
// }

// TEST(PIRTest, NotEqualConstraint3) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var 1..10: x; constraint bool_not(int_eq(x, 10));", env);
//   deduce_and_test(pir, 1, {Itv(1,10)}, {Itv(1,9)}, true);
// }

// // Constraint of the form "a[b] = c".
// TEST(PIRTest, ElementConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>(
//     "array[1..3] of int: a = [10, 11, 12];\
//     var 1..3: b; var 10..12: c;\
//     constraint array_int_element(b, a, c);", env);
//   deduce_and_test(pir, 3, {Itv(1,3), Itv(10, 12)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(c, 11);", pir, env);
//   deduce_and_test(pir, 3, {Itv(1,3), Itv(10, 11)}, {Itv(1,2), Itv(10,11)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_ge(c, 11);", pir, env);
//   deduce_and_test(pir, 3, {Itv(1,2), Itv(11, 11)}, {Itv(2,2), Itv(11,11)}, true);
// }


// // Constraint of the form "x = 5 xor y = 5".
// TEST(PIRTest, XorConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var int: x; var int: y;\
//     constraint bool_xor(int_eq(x, 5), int_eq(y, 5));", env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv::top()}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(1, 1), Itv::top()}, {Itv(1, 1), Itv(5, 5)}, true);
// }

// // Constraint of the form "x = 5 xor y = 5".
// TEST(PIRTest, XorConstraint2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = create_and_interpret_and_tell<IPIR>("var 1..5: x; var 1..5: y;\
//     constraint bool_xor(int_eq(x, 5), int_eq(y, 5));", env);
//   deduce_and_test(pir, 1, {Itv(1, 5), Itv(1, 5)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y, 5);", pir, env);
//   deduce_and_test(pir, 1, {Itv(1, 5), Itv(5, 5)}, {Itv(1, 4), Itv(5, 5)}, true);
// }

// template <class F>
// void type_in_predicate(F& f, AType ty) {
//   switch(f.index()) {
//     case F::Seq:
//       if(f.sig() == IN && f.seq(1).is(F::S) && f.seq(1).s().size() > 1) {
//         f.type_as(ty);
//         return;
//       }
//       for(int i = 0; i < f.seq().size(); ++i) {
//         type_in_predicate(f.seq(i), ty);
//       }
//       break;
//     case F::ESeq:
//       for(int i = 0; i < f.eseq().size(); ++i) {
//         type_in_predicate(f.eseq(i), ty);
//       }
//       break;
//   }
// }

// template <class L>
// L interpret_type_and_tell(const char* fzn, VarEnv<standard_allocator>& env) {
//   return create_and_interpret_and_type_and_tell<L>(fzn, env, [](auto& f) { type_in_predicate(f, 1); });
// }

// // Constraint of the form "x in {1,3}".
// TEST(PIRTest, InConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var {1, 3}: x; var 2..3: y;", env);
//   deduce_and_test(pir, 1, {Itv(1, 3), Itv(2,3)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, y);", pir, env);
//   deduce_and_test(pir, 2, {Itv(1, 3), Itv(2, 3)}, {Itv(3,3), Itv(3,3)}, true);
// }

// // min(x, y) = z
// TEST(PIRTest, MinConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var 0..4: x; var 2..5: y; var 0..10: z;\
//     constraint int_min(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(0, 10)}, {Itv(0, 4), Itv(2, 5), Itv(0, 4)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(z, 3);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(0, 3)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(x, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(2, 5), Itv(0, 3)}, {Itv(0, 1), Itv(2, 5), Itv(0, 1)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(x, 0);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(2, 5), Itv(0, 1)}, {Itv(0, 0), Itv(2, 5), Itv(0, 0)}, true);
// }

// // min(x, y) = z
// TEST(PIRTest, MinConstraint2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var 0..4: x; var 2..5: y; var 0..10: z;\
//     constraint int_min(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(0, 10)}, {Itv(0, 4), Itv(2, 5), Itv(0, 4)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(z, 3);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(0, 3)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 4);", pir, env);
//   deduce_and_test(pir, 1, {Itv(4, 4), Itv(2, 5), Itv(0, 3)}, {Itv(4, 4), Itv(2, 3), Itv(2, 3)}, false);
// }

// // min(x, y) = z
// TEST(PIRTest, MinConstraint3) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var int: x; var 0..1: b1; var 0..1: b2;\
//     constraint int_min(b1, b2, 1);", env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv(0, 1), Itv(0, 1)}, {Itv::top(), Itv(1,1), Itv(1, 1)}, true);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b1, int_le(x, 5));", pir, env);
//   deduce_and_test(pir, 2, {Itv::top(), Itv(1,1), Itv(1, 1)}, {Itv(Itv::LB::top(), 5), Itv(1, 1), Itv(1, 1)}, true);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b2, int_ge(x, 5));", pir, env);
//   deduce_and_test(pir, 3, {Itv(Itv::LB::top(), 5), Itv(1, 1), Itv(1, 1)}, {Itv(5, 5), Itv(1, 1), Itv(1, 1)}, true);
// }

// // max(x, y) = z
// TEST(PIRTest, MaxConstraint1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var 0..4: x; var 2..5: y; var 0..10: z;\
//     constraint int_max(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(0, 10)}, {Itv(0, 4), Itv(2, 5), Itv(2, 5)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(z, 3);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(2, 3)}, {Itv(0, 3), Itv(2, 3), Itv(2, 3)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_le(x, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(2, 3), Itv(2, 3)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y, 2);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(2, 2), Itv(2, 3)}, {Itv(0, 1), Itv(2, 2), Itv(2, 2)}, true);
// }

// // max(x, y) = z
// TEST(PIRTest, MaxConstraint2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var 0..4: x; var 2..5: y; var 0..10: z;\
//     constraint int_max(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(0, 10)}, {Itv(0, 4), Itv(2, 5), Itv(2, 5)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_ge(z, 5);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 4), Itv(2, 5), Itv(5, 5)}, {Itv(0, 4), Itv(5, 5), Itv(5, 5)}, true);
// }

// TEST(PIRTest, MaxConstraint3) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("var int: x; var 0..1: b1; var 0..1: b2;\
//     constraint int_max(b1, b2, 0);", env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv(0, 1), Itv(0, 1)}, {Itv::top(), Itv(0,0), Itv(0,0)}, true);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b1, int_le(x, 5));", pir, env);
//   deduce_and_test(pir, 2, {Itv::top(), Itv(0,0), Itv(0,0)}, {Itv(6, Itv::UB::top()), Itv(0, 0), Itv(0, 0)}, true);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b2, int_ge(x, 7));", pir, env);
//   deduce_and_test(pir, 3, {Itv(6, Itv::UB::top()), Itv(0, 0), Itv(0, 0)}, {Itv(6, 6), Itv(0, 0), Itv(0, 0)}, true);
// }

// TEST(PIRTest, BooleanClause1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("array[1..2] of var bool: x;\
//     array[1..2] of var bool: y;\
//     constraint bool_clause(x, y);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x[1], true);", pir, env);
//   deduce_and_test(pir, 1, {Itv(1, 1), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, true);
// }

// TEST(PIRTest, BooleanClause2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("array[1..2] of var bool: x;\
//     array[1..2] of var bool: y;\
//     constraint bool_clause(x, y);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y[1], false);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 0), Itv(0, 1)}, true);
// }

// TEST(PIRTest, BooleanClause3) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("array[1..2] of var bool: x;\
//     array[1..2] of var bool: y;\
//     constraint bool_clause(x, y);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x[1], false);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x[2], false);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 0), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y[1], true);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 0), Itv(1, 1), Itv(0, 1)},  {Itv(0, 0), Itv(0, 0), Itv(1, 1), Itv(0, 0)}, true);
// }

// TEST(PIRTest, BooleanClause4) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("array[1..2] of var bool: x;\
//     array[1..2] of var bool: y;\
//     constraint bool_clause(x, y);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x[1], false);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y[1], true);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 1), Itv(1, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y[2], true);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 1), Itv(1, 1), Itv(1, 1)},  {Itv(0, 0), Itv(1, 1), Itv(1, 1), Itv(1, 1)}, true);
// }

// TEST(PIRTest, IntTimes1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var 0..1: x;\
//     var 0..1: y;\
//     var 0..1: z;\
//     constraint int_times(x,y,z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(1, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(1, 1), Itv(1, 1), Itv(0, 1)}, {Itv(1, 1), Itv(1, 1), Itv(1, 1)}, true);
// }

// TEST(PIRTest, IntTimes2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var 0..1: x;\
//     var 0..1: y;\
//     var 0..1: z;\
//     constraint int_times(x,y,z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(0, 1)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(z, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 1), Itv(1, 1)}, {Itv(1, 1), Itv(1, 1), Itv(1, 1)}, true);
// }

// TEST(PIRTest, IntTimes3) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var 0..0: x; \
//     var 0..1: y; \
//     var 0..1: z; \
//     constraint int_times(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 0), Itv(0, 1), Itv(0, 1)}, {Itv(0, 0), Itv(0, 1), Itv(0, 0)}, true);
// }

// TEST(PIRTest, IntTimes4) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var 0..1: x; \
//     var 0..0: y; \
//     var 0..1: z; \
//     constraint int_times(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(0, 0), Itv(0, 1)}, {Itv(0, 1), Itv(0, 0), Itv(0, 0)}, true);
// }

// TEST(PIRTest, IntTimes5) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var 1..2: x; \
//     var 0..1: y; \
//     var 0..0: z; \
//     constraint int_times(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(1, 2), Itv(0, 1), Itv(0, 0)}, {Itv(1, 2), Itv(0, 0), Itv(0, 0)}, true);
// }

// TEST(PIRTest, IntTimes6) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var 0..1: x; \
//     var 1..2: y; \
//     var 0..0: z; \
//     constraint int_times(x, y, z);", env);
//   deduce_and_test(pir, 1, {Itv(0, 1), Itv(1, 2), Itv(0, 0)}, {Itv(0, 0), Itv(1, 2), Itv(0, 0)}, true);
// }

// TEST(PIRTest, IntAbs1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var -15..5: x;\
//     var -10..10: y;\
//     constraint int_abs(x, y);", env);
//   deduce_and_test(pir, 1, {Itv(-15, 5), Itv(-10, 10)}, {Itv(-10, 5), Itv(0, 10)}, false);
// }

// template <class L>
// L interpret_type_and_tell2(const char* fzn, VarEnv<standard_allocator>& env) {
//   return create_and_interpret_and_type_and_tell<L>(fzn, env, [](auto& f) { f.seq(0).type_as(sty); f.seq(1).type_as(sty); });
// }

// TEST(PIRTest, AbstractElement1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell2<IPIR>("\
//     var 0..10: x;\
//     var 0..10: y;\
//     constraint nbool_equiv(int_ge(x, 5), int_le(y, 5));", env);
//   deduce_and_test(pir, 1, {Itv(0, 10), Itv(0, 10)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 5);", pir, env);
//   deduce_and_test(pir, 1, {Itv(5, 5), Itv(0, 10)}, {Itv(5, 5), Itv(0, 5)}, true);
// }

// TEST(PIRTest, AbstractElement2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell2<IPIR>("\
//     var 0..10: x;\
//     var 0..10: y;\
//     constraint nbool_equiv(int_ge(x, 5), int_le(y, 5));", env);
//   deduce_and_test(pir, 1, {Itv(0, 10), Itv(0, 10)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 4);", pir, env);
//   deduce_and_test(pir, 1, {Itv(4, 4), Itv(0, 10)}, {Itv(4, 4), Itv(6, 10)}, true);
// }

// TEST(PIRTest, AbstractElement3) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell2<IPIR>("\
//     var 0..10: x;\
//     var 0..10: y;\
//     constraint nbool_equiv(int_eq(x, 5), int_eq(y, 5));", env);
//   deduce_and_test(pir, 1, {Itv(0, 10), Itv(0, 10)}, false);
//   // As interval does not support hole, we cannot reduce the domain of `x`.
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y, 6);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 10), Itv(6, 6)}, false);
//   // In that case, the propagation is weaker but the result is still correct, this is not always the case, see below.
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 5);", pir, env);
//   deduce_and_test(pir, 1, {Itv(5, 5), Itv(6, 6)}, {Itv(5, 5), Itv::bot()}, false, true);
// }

// TEST(PIRTest, AbstractElement4) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell2<IPIR>("\
//     var 0..10: x;\
//     var 0..10: y;\
//     constraint nbool_equiv(int_eq(x, 5), int_eq(y, 5));", env);
//   deduce_and_test(pir, 1, {Itv(0, 10), Itv(0, 10)}, false);
//   // As interval does not support hole, we cannot reduce the domain of `x`.
//   // Further, because y = 5 is going to be detected as false, and we will thus push x != 5, which does not reduce any domain, so what we obtain is not a solution (it is still an over-approximation though).
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(y, 4);", pir, env);
//   deduce_and_test(pir, 1, {Itv(0, 10), Itv(4, 4)}, false);
//   interpret_must_succeed<IKind::TELL>("constraint int_eq(x, 5);", pir, env);
//   deduce_and_test(pir, 1, {Itv(5, 5), Itv(4, 4)}, {Itv(5,5), Itv(5, 4)}, false, true);
// }

// TEST(PIRTest, InfiniteDomain1) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var int: x; \
//     var bool: b; \
//     constraint int_eq(b, int_le(x,5));", env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv(0,1)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b, 1);", pir, env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv(1,1)}, {Itv(Itv::LB::top(), 5), Itv(1,1)}, true);
// }

// TEST(PIRTest, InfiniteDomain2) {
//   VarEnv<standard_allocator> env;
//   IPIR pir = interpret_type_and_tell<IPIR>("\
//     var int: x; \
//     var bool: b; \
//     constraint int_eq(b, int_le(x,5));", env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv(0,1)}, false);

//   interpret_must_succeed<IKind::TELL>("constraint int_eq(b, 0);", pir, env);
//   deduce_and_test(pir, 1, {Itv::top(), Itv(0,0)}, {Itv(6, Itv::UB::top()), Itv(0,0)}, true);
// }
