// Copyright 2025 Yi-Nung Tsao

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"
#include "bound_consistency_test.hpp"

#include "battery/vector.hpp"
#include "battery/shared_ptr.hpp"

#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/pir.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"

#include <format>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace lala;
using namespace battery;

using F = TFormula<standard_allocator>;

using FStore = VStore<FItv, standard_allocator>;
using FPIR = PIR<FStore>; // Floating Interval Propagators Completion

const AType sty = 0;
const AType pty = 1;

template <class L>
void test_extract(const L& fpir, bool is_ua) {
  AbstractDeps<standard_allocator> deps(standard_allocator{});
  L copy1(fpir, deps);
  if(is_ua) {
    for(int i = 0; i < fpir.vars(); ++i) {
      printf("fpir[%d] = [%f,%f]\n", i, fpir[i].lb().value(), fpir[i].ub().value());
    }
    for(int i = 0; i < fpir.num_deductions(); ++i) {
      EXPECT_TRUE(fpir.ask(i)) << "fpir.ask(" << i << ") == false";
    }
  }
  EXPECT_EQ(fpir.is_extractable(), is_ua);
  if(fpir.is_extractable()) {
    fpir.extract(copy1);
    EXPECT_EQ(fpir.is_top(), copy1.is_top());
    EXPECT_EQ(fpir.is_bot(), copy1.is_bot());
    for(int i = 0; i < fpir.vars(); ++i) {
      EXPECT_EQ(fpir[i], copy1[i]);
    }
  }
}

template<class L>
void deduce_and_test(L& fpir, int num_deds, const std::vector<FItv>& before, const std::vector<FItv>& after, bool is_ua) {
  EXPECT_EQ(fpir.num_deductions(), num_deds);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(fpir[i], before[i]) << "fpir[" << i << "]";
  }
  GaussSeidelIteration{}.fixpoint(
    fpir.num_deductions(),
    [&](size_t i) { return fpir.fdeduce(i); });
  /** Note: We don't test has_changed anymore due to the internal variable, it usually changes due to the unbounded domains of the internal variables. */
  for(int i = 0; i < after.size(); ++i) {
    std::cout << "fpir[" << i << "]" << std::setprecision(std::numeric_limits<double>::max_digits10) << fpir[i].lb().value() << ", " << fpir[i].ub().value() << std::endl;
    std::cout << "after[" << i << "]" << std::setprecision(std::numeric_limits<double>::max_digits10) << after[i].lb().value() << ", " << after[i].ub().value() << std::endl;
    EXPECT_EQ(fpir[i], after[i]) << "fpir[" << i << "]";
  }
  test_extract(fpir, is_ua);
}

template<class L>
void deduce_and_test(L& fpir, int num_deds, const std::vector<FItv>& before_after, bool is_ua = false) {
  deduce_and_test(fpir, num_deds, before_after, before_after, is_ua);
}

template<class L>
void deduce_and_test_bot(L& fpir, int num_deds, const std::vector<FItv>& before) {
  EXPECT_EQ(fpir.num_deductions(), num_deds);
  for(int i = 0; i < before.size(); ++i) {
    EXPECT_EQ(fpir[i], before[i]) << "fpir[" << i << "]";
  }
  local::B has_changed = false;
  GaussSeidelIteration{}.fixpoint(
    fpir.num_deductions(),
    [&](size_t i) { return fpir.fdeduce(i); },
    has_changed
  );
  EXPECT_TRUE(has_changed);
  EXPECT_TRUE(fpir.is_bot());
}

#ifdef NDEBUG

TEST(FPIRTest, TernaryPropagatorSoundnessTest) {
  test_bound_propagator_soundness<FPIR>("float_eq_reif", [](float x, float y, float z) { return (x == 0 || x == 1) && x == (y == z); }, true, true);
  test_bound_propagator_soundness<FPIR>("float_le_reif", [](float x, float y, float z) { return (x == 0 || x == 1) && x == (y <= z); }, true, true);
  test_bound_propagator_soundness<FPIR>("float_plus", [](float x, float y, float z) { return x == y + z; });
  test_bound_propagator_soundness<FPIR>("float_min", [](float x, float y, float z) { return x == std::min(y, z); });
  test_bound_propagator_soundness<FPIR>("float_max", [](float x, float y, float z) { return x == std::max(y, z); });
  test_bound_propagator_soundness<FPIR>("float_times", [](float x, float y, float z) { return x == y * z; }, false);
}

TEST(FPIRTest, TernaryPropagatorCompletenessTest) {
  test_bound_propagator_completeness<FPIR>("float_eq_reif", [](float x, float y, float z) { return (x == 0 || x == 1) && x == (y == z); });
  test_bound_propagator_completeness<FPIR>("float_le_reif", [](float x, float y, float z) { return (x == 0 || x == 1) && x == (y <= z); });
  test_bound_propagator_completeness<FPIR>("float_plus", [](float x, float y, float z) { return x == y + z; });
  test_bound_propagator_completeness<FPIR>("float_min", [](float x, float y, float z) { return x == std::min(y, z); });
  test_bound_propagator_completeness<FPIR>("float_max", [](float x, float y, float z) { return x == std::max(y, z); });
  test_bound_propagator_completeness<FPIR>("float_times", [](float x, float y, float z) { return x == y * z; });
}

#endif

TEST(FPIRTest, TernaryProblem) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_ge(z, 5.0); constraint float_le(z, 5.0);\
    constraint float_plus(x, y, z);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0), FItv(5.0,5.0)}, {FItv(0.0,5.0), FItv(0.0,5.0), FItv(5.0,5.0)}, false);
}

TEST(FPIRTest, ReifiedEquality) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_eq(x, float_eq(y,2.0));", false);
  deduce_and_test(fpir, 1, {FItv(0.0,1.0), FItv::top()}, {FItv(0.0,1.0), FItv::top()}, false);
}

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// TEST(FPIRtest, reifiedin) {
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var {1.0,2.0,4.0,5.0}: x; var bool: y;\
//     constraint set_in_reif(x, 2.0..2.0, y);");
//   /* Generated variables:
//   ((var x:z):-1 ∧
//   (var __constant_1:z):-1 ∧
//   (var __var_b_0:b):-1 ∧
//   (var __constant_2:z):-1 ∧
//   (var __var_b_1:b):-1 ∧
//   (var __var_b_2:b):-1 ∧
//   (var __constant_4:z):-1 ∧
//   (var __var_b_3:b):-1 ∧
//   (var __constant_5:z):-1 ∧
//   (var __var_b_4:b):-1 ∧
//   (var __var_b_5:b):-1 ∧
//   (var y:b):-1 ∧
//   */
//   deduce_and_test(fpir, 8,
//     {FItv(1.0,5.0), FItv(1.0,1.0), FItv(0.0,1.0), FItv(2.0,2.0), FItv(0.0,1.0), FItv(0.0,1.0), FItv(4.0,4.0), FItv(0.0,1.0), FItv(5.0,5.0), FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)},
//     {FItv(1.0,5.0), FItv(1.0,1.0), FItv(1.0,1.0), FItv(2.0,2.0), FItv(0.0,1.0), FItv(0.0,1.0), FItv(4.0,4.0), FItv(0.0,1.0), FItv(5.0,5.0), FItv(1.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)},
//     false);
// }

// x + y = 5.0
TEST(FPIRTest, AddEquality) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_plus(x, y, 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)}, {FItv(0.0,5.0), FItv(0.0,5.0)}, false);
}

// x + y = z, z <= 5.0
TEST(FPIRTest, TemporalConstraint1Flat) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_le(z, 5.0);\
    constraint float_plus(x, y, z);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0), FItv(flb::top(), fub(5.0))}, {FItv(0.0,5.0), FItv(0.0,5.0), FItv(0.0,5.0)}, false);
}

TEST(FPIRTest, TemporalConstraint1) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_le(float_plus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)}, {FItv(0.0,5.0), FItv(0.0,5.0)}, false);
}

// x + y >= 5.0 (x,y in [0.0..10.0])
TEST(FPIRTest, TemporalConstraint2) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_ge(float_plus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)});
}

// x + y >= 5.0 (x,y in [0.0..3.0])
TEST(FPIRTest, TemporalConstraint3) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 3.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 3.0);\
    constraint float_ge(float_plus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,3.0), FItv(0.0,3.0)}, {FItv(2.0,3.0), FItv(2.0,3.0)}, false);
}

// x + y >= 5.0 (x,y in [0.0..3.0])
TEST(FPIRTest, TemporalConstraint4) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 3.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 3.0);\
    constraint float_ge(float_plus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,3.0), FItv(0.0,3.0)}, {FItv(2.0,3.0), FItv(2.0,3.0)}, false);
}

// x + y = 5.0 (x,y in [0.0..4.0])
TEST(FPIRTest, TemporalConstraint5) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 4.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 4.0);\
    constraint float_eq(float_plus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,4.0), FItv(0.0,4.0)}, {FItv(1.0,4.0), FItv(1.0,4.0)}, false);
}

// x - y <= 5.0 (x,y in [0.0..10.0])
TEST(FPIRTest, TemporalConstraint6) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_le(float_minus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)});
}

// x - y <= -10.0 (x,y in [0.0..10.0])
TEST(FPIRTest, TemporalConstraint7) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_le(float_minus(x, y), -10.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)}, {FItv(0.0,0.0), FItv(10.0,10.0)}, true);
}

// x - y >= 5.0 (x,y in [0.0..10.0])
TEST(FPIRTest, TemporalConstraint8) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_ge(float_minus(x, y), 5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)}, {FItv(5.0,10.0), FItv(0.0,5.0)}, false);
}

// x - y <= -5.0 (x,y in [0.0..10.0])
TEST(FPIRTest, TemporalConstraint9) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_le(float_minus(x, y), -5.0);");
  deduce_and_test(fpir, 1, {FItv(0.0,10.0), FItv(0.0,10.0)}, {FItv(0.0,5.0), FItv(5.0,10.0)}, false);
}

// x <= -5.0 + y (x,y in [0.0..10.0])
TEST(FPIRTest, TemporalConstraint10) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
    constraint float_ge(x, 0.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 10.0);\
    constraint float_le(x, float_plus(-5.0,y));");
  deduce_and_test(fpir, 2, {FItv(0.0,10.0), FItv(0.0,10.0)}, {FItv(0.0,5.0), FItv(5.0,10.0)}, false);
}

// TOP test x,y,z in [3..10] /\ x + y + z <= 8
TEST(FPIRTest, TopProp) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 3.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 3.0); constraint float_le(y, 10.0);\
    constraint float_ge(z, 3.0); constraint float_le(z, 10.0);\
    constraint float_le(float_plus(float_plus(x, y),z), 8.0);");
  deduce_and_test_bot(fpir, 2, {FItv(3.0,10.0), FItv(3.0,10.0), FItv(3.0,10.0)});
}

// x,y,z in [3.0..10.0] /\ x + y + z <= 9.0
TEST(FPIRTest, TernaryAdd1) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 3.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 3.0); constraint float_le(y, 10.0);\
    constraint float_ge(z, 3.0); constraint float_le(z, 10.0);\
    constraint float_le(float_plus(float_plus(x, y),z), 9.0);");
  deduce_and_test(fpir, 2, {FItv(3.0,10.0), FItv(3.0,10.0), FItv(3.0,10.0)}, {FItv(3.0,3.0), FItv(3.0,3.0), FItv(3.0,3.0)}, true);
}

// x,y,z in [3.0..10.0] /\ x + y + z <= 10.0
TEST(FPIRTest, TernaryAdd2) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 3.0); constraint float_le(x, 10.0);\
    constraint float_ge(y, 3.0); constraint float_le(y, 10.0);\
    constraint float_ge(z, 3.0); constraint float_le(z, 10.0);\
    constraint float_le(float_plus(float_plus(x, y),z), 10.0);");
  deduce_and_test(fpir, 2, {FItv(3.0,10.0), FItv(3.0,10.0), FItv(3.0,10.0)}, {FItv(3.0,4.0), FItv(3.0,4.0), FItv(3.0,4.0)}, false);
}

// x,y,z in [-2.0..2.0] /\ x + y + z <= -5.0
TEST(FPIRTest, TernaryAdd3) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, {FItv(-2.0,2.0), FItv(-2.0,2.0), FItv(-2.0,2.0)}, {FItv(-2.0,-1.0), FItv(-2.0,-1.0), FItv(-2.0,-1.0)}, false);
}

TEST(FPIRTest, TernaryAdd4) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.123456); constraint float_le(x, 2.123456);\
    constraint float_ge(y, -2.123456); constraint float_le(y, 2.123456);\
    constraint float_ge(z, -2.123456); constraint float_le(z, 2.123456);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.77321);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("-2.123456","2.123456"), create_float_interval<FItv>("-2.123456","2.123456"), create_float_interval<FItv>("-2.123456","2.123456")}, 
    {create_float_interval<FItv>("-2.123456","-1.52629"), create_float_interval<FItv>("-2.123456","-1.52629"), create_float_interval<FItv>("-2.123456","-1.52629")}, false);
}

TEST(FPIRTest, TernaryAdd5) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

TEST(FPIRTest, TernaryAdd6) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

TEST(FPIRTest, TernaryAdd7) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

TEST(FPIRTest, TernaryMul1) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

TEST(FPIRTest, TernaryMul2) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false); 
}

TEST(FPIRTest, TernaryMul3) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

TEST(FPIRTest, TernaryMul4) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

TEST(FPIRTest, TernaryMul5) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, -2.0); constraint float_le(x, 2.0);\
    constraint float_ge(y, -2.0); constraint float_le(y, 2.0);\
    constraint float_ge(z, -2.0); constraint float_le(z, 2.0);\
    constraint float_le(float_plus(float_plus(x, y),z), -5.0);");
  deduce_and_test(fpir, 2, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, 
    {create_float_interval<FItv>("",""), create_float_interval<FItv>("",""), create_float_interval<FItv>("","")}, false);
}

// x,y,z in [0.0..1.0] /\ 2.0x + y + 4.0z <= 2.0
TEST(FPIRTest, PseudoBoolean1) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 0.0); constraint float_le(x, 1.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 1.0);\
    constraint float_ge(z, 0.0); constraint float_le(z, 1.0);\
    constraint float_le(float_plus(float_plus(float_times(2.0,x), y),float_times(4.0,z)), 2.0);");
  deduce_and_test(fpir, 4, {FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)}, {FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,0.5)}, false);
}

// x,y,z in [0.0..1.0] /\ 2.0x + 1.0y + 2.0z <= 2.0
TEST(FPIRTest, PseudoBoolean2) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 0.0); constraint float_le(x, 1.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 1.0);\
    constraint float_ge(z, 0.0); constraint float_le(z, 1.0);\
    constraint float_le(float_plus(float_plus(float_times(2.0,x), float_times(1.0,y)),float_times(2.0,z)), 2.0);");
  deduce_and_test(fpir, 5, {FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)}, {FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)}, false);
}

// x,y,z in [0.0..1.0] /\ 4.0x + 2.0y + 4.0z <= 2.0
TEST(FPIRTest, PseudoBoolean3) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 0.0); constraint float_le(x, 1.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 1.0);\
    constraint float_ge(z, 0.0); constraint float_le(z, 1.0);\
    constraint float_le(float_plus(float_plus(float_times(4.0,x), float_times(2.0,y)),float_times(4.0,z)), 2.0);");
  deduce_and_test(fpir, 5, {FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)}, {FItv(0.0,0.5), FItv(0.0,1.0), FItv(0.0,0.5)}, false);
}

// x,y,z in [0.0..1.0] /\ -x + y + 3z <= 2.0
TEST(FPIRTest, PseudoBoolean4) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var float: z;\
    constraint float_ge(x, 0.0); constraint float_le(x, 1.0);\
    constraint float_ge(y, 0.0); constraint float_le(y, 1.0);\
    constraint float_ge(z, 0.0); constraint float_le(z, 1.0);\
    constraint float_le(float_plus(float_plus(float_neg(x), y), float_times(3.0,z)), 2.0);");
  deduce_and_test(fpir, 4, {FItv(0.0,1.0), FItv(0.0,1.0), FItv(0.0,1.0)}, false);
}

// x in [-4.0..3.0], -x <= 2.0
TEST(FPIRTest, NegationOp1) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, 2.0);\
    constraint float_ge(x, -4.0); constraint float_le(x, 3.0);\
    constraint float_le(float_neg(x), y);");
  deduce_and_test(fpir, 2, {FItv(-4.0,3.0), FItv(2.0,2.0)}, {FItv(-2.0,3.0), FItv(2.0,2.0)}, false);
}

// x in [-4.0..3.0], -x <= -2.0
TEST(FPIRTest, NegationOp2) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, -2.0);\
    constraint float_ge(x, -4.0); constraint float_le(x, 3.0);\
    constraint float_le(float_neg(x), y);");
  deduce_and_test(fpir, 2, {FItv(-4.0,3.0), FItv(-2.0,-2.0)}, {FItv(2.0,3.0), FItv(-2.0,-2.0)}, false);
}

// x in [0.0..3.0], -x <= -2.0
TEST(FPIRTest, NegationOp3) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, -2.0);\
    constraint float_ge(x, 0.0); constraint float_le(x, 3.0);\
    constraint float_le(float_neg(x), y);");
  deduce_and_test(fpir, 2, {FItv(0.0,3.0), FItv(-2.0,-2.0)}, {FItv(2.0,3.0), FItv(-2.0,-2.0)}, false);
}

// x in [-4.0..-3.0], -x <= 4.0
TEST(FPIRTest, NegationOp4) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, 4.0);\
    constraint float_ge(x, -4.0); constraint float_le(x, -3.0);\
    constraint float_le(float_neg(x), y);");
  deduce_and_test(fpir, 2, {FItv(-4.0,-3.0), FItv(4.0, 4.0)}, false);
}

// x in [-4.0..3.0], -x >= -2.0
TEST(FPIRTest, NegationOp5) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, -2.0);\
    constraint float_ge(x, -4.0); constraint float_le(x, 3.0);\
    constraint float_ge(float_neg(x), y);");
  deduce_and_test(fpir, 2, {FItv(-4.0,3.0), FItv(-2.0,-2.0)}, {FItv(-4.0,2.0), FItv(-2.0,-2.0)}, false);
}

// x in [-4.0..3.0], -x > 2.0
TEST(FPIRTest, NegationOp6) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, 2.0);\
    constraint float_ge(x, -4.0); constraint float_le(x, 3.0);\
    constraint float_ge(float_neg(x), y);");
  deduce_and_test(fpir, 2, {FItv(-4.0,3.0), FItv(2.0,2.0)}, {FItv(-4.0,-2.0), FItv(2.0,2.0)}, false);
}

// x in [-4.0..3.0], -x >= 5.0
TEST(FPIRTest, NegationOp7) {
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x;\
    var float: y;\
    constraint float_eq(y, 5.0);\
    constraint float_ge(x, -4.0); constraint float_le(x, 3.0);\
    constraint float_ge(float_neg(x), y);");
  deduce_and_test_bot(fpir, 2, {FItv(-4.0,3.0), FItv(5.0,5.0)});
}

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// Constraint of the form "b <=> (x - y <= k1 /\ y - x <= k2)".
// TEST(FPIRTest, ResourceConstraint1) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var int: b;\
//     constraint float_ge(x, 5.0); constraint float_le(x, 10.0);\
//     constraint float_ge(y, 9.0); constraint float_le(y, 15.0);\
//     constraint int_ge(b, 0); constraint int_le(b, 1);\
//     constraint int_eq(b, bool_and(float_le(float_minus(x, y), 0.0), float_le(float_minus(y, x), 2.0)));", env);
//   deduce_and_test(fpir, 5, {FItv(5.0,10.0), FItv(9.0,15.0), Itv(0,1)}, false);

//   interpret_must_succeed<IKind::TELL, false, false>("constraint int_eq(b, 1);", fpir, env);
//   deduce_and_test(fpir, 5, {FItv(5.0,10.0), FItv(9.0,15.0), FItv(1,1)}, {FItv(7.0,10.0), FItv(9.0,12.0), FItv(1,1)}, false);
// }

// TEST(FPIRTest, ResourceConstraint2) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y; var int: b;\
//     constraint float_ge(x, 1.0); constraint float_le(x, 2.0);\
//     constraint float_ge(y, 0.0); constraint float_le(y, 2.0);\
//     constraint int_ge(b, 0); constraint int_le(b, 1);\
//     constraint int_eq(b, bool_and(float_le(float_minus(x, y), 2.0), float_le(float_minus(y, x), -1.0)));", env);
//   deduce_and_test(fpir, 5, {FItv(1.0,2.0), FItv(0.0,2.0), FItv(0,1)}, false);

//   interpret_must_succeed<IKind::TELL, false, false>("constraint int_eq(b, 0);", fpir, env);
//   deduce_and_test(fpir, 5, {FItv(1.0,2.0), FItv(0.0,2.0), FItv(0,0)}, {FItv(1.0,2.0), FItv(1.0,2.0), FItv(0,0)}, false);
// }

TEST(FPIRTest, Strict1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 10.0..10.0: y; constraint float_gt(x, y);", env);
  deduce_and_test_bot(fpir, 1, {FItv(1.0,10.0)});
}

TEST(FPIRTest, Strict2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 10.0..10.0: y; constraint float_lt(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0)}, {FItv(1.0,9.0)}, true);
}

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// TEST(FPIRTest, Strict3) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 10.0..10.0: y; constraint nbool_or(float_lt(x, y), float_gt(x, y));", env);
//   deduce_and_test(fpir, 5, {FItv(1.0,10.0)}, {FItv(1.0,9.0)}, true);
// }

TEST(FPIRTest, EqualConstraint1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 9.0..10.0: y; constraint float_eq(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0), FItv(9.0,10.0)}, {FItv(9.0,10.0), FItv(9.0,10.0)}, false);
}

TEST(FPIRTest, EqualConstraint2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 1.0..2.0: y; constraint float_eq(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0), FItv(1.0,2.0)}, {FItv(1.0,2.0), FItv(1.0,2.0)}, false);
}

TEST(FPIRTest, EqualConstraint3) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 0.0..11.0: y; constraint float_eq(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0), FItv(0.0,11.0)}, {FItv(1.0,10.0), FItv(1.0,10.0)}, false);
}

TEST(FPIRTest, EqualConstraint4) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 5.0..11.0: y; constraint float_eq(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0), FItv(5.0,11.0)}, {FItv(5.0,10.0), FItv(5.0,10.0)}, false);
}

TEST(FPIRTest, NotEqualConstraint1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; constraint float_ne(x, 10.0);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0)}, {FItv(1.0,10.0)}, false);
}

TEST(FPIRTest, NotEqualConstraint2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 10.0..10.0: y; constraint float_ne(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0), FItv(10.0,10.0)}, {FItv(1.0,9.0), FItv(10.0,10.0)}, false);
}

TEST(FPIRTest, NotEqualConstraint3) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; var 1.0..1.0: y; constraint float_ne(x, y);", env);
  deduce_and_test(fpir, 1, {FItv(1.0,10.0), FItv(1.0,1.0)}, {FItv(1.0,10.0), FItv(1.0,1.0)}, false);
}

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// TEST(FPIRTest, NotEqualConstraint4) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..10.0: x; constraint bool_not(float_eq(x, 10.0));", env);
//   deduce_and_test(fpir, 1, {FItv(1.0,10.0)}, {FItv(1.0,9.0)}, false);
// }

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// Constraint of the form "a[b] = c".
// TEST(FPIRTest, ElementConstraint1) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>(
//     "array[1..3] of float: a = [10.0, 11.0, 12.0];\
//     var 1..3: b; var 10.0..12.0: c;\
//     constraint array_int_element(b, a, c);", env);
//   deduce_and_test(fpir, 12, {FItv(1.0,3.0), FItv(10.0, 12.0)}, false);

//   interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(c, 11.0);", fpir, env);
//   deduce_and_test(fpir, 12, {FItv(1.0,3.0), FItv(10.0, 11.0)}, {FItv(1.0,2.0), FItv(10.0,11.0)}, false);

//   interpret_must_succeed<IKind::TELL, false, false>("constraint float_ge(c, 11.0);", fpir, env);
//   deduce_and_test(fpir, 12, {FItv(1.0,2.0), FItv(11.0, 11.0)}, {FItv(2.0,2.0), FItv(11.0,11.0)}, true);
// }

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// // Constraint of the form "x = 5.0 xor y = 5.0". 
// TEST(FPIRTest, XorConstraint1) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var float: y;\
//     constraint bool_xor(float_eq(x, 5.0), float_eq(y, 5.0));", env);
//   deduce_and_test(fpir, 3, {FItv::top(), FItv::top()}, false);

//   interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(x, 1.0);", fpir, env);
//   deduce_and_test(fpir, 3, {FItv(1.0, 1.0), FItv::top()}, {FItv(1.0, 1.0), FItv(5.0, 5.0)}, true);
// }

// // Constraint of the form "x = 5.0 xor y = 5.0".
// TEST(FPIRTest, XorConstraint2) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 1.0..5.0: x; var 1.0..5.0: y;\
//     constraint bool_xor(float_eq(x, 5.0), float_eq(y, 5.0));", env);
//   deduce_and_test(fpir, 3, {FItv(1.0, 5.0), FItv(1.0, 5.0)}, false);

//   interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(y, 5.0);", fpir, env);
//   deduce_and_test(fpir, 3, {FItv(1.0, 5.0), FItv(5.0, 5.0)}, {FItv(1.0, 4.0), FItv(5.0, 5.0)}, true);
// }

// The domain is mixing with floating point and boolean. 
// It is not supported yet, so skip to test for now. 
//
// Constraint of the form "x in {1.0,3.0}".
// TEST(FPIRTest, InConstraint1) {
//   VarEnv<standard_allocator> env;
//   FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var {1.0, 3.0}: x; var 2.0..3.0: y;", env);
//   /* Generated variables:
//   ((var x:Z):-1 ∧
//   (var __CONSTANT_1:Z):-1 ∧
//   (var __VAR_B_0:B):-1 ∧
//   (var __CONSTANT_3:Z):-1 ∧
//   (var __VAR_B_1:B):-1 ∧
//   (var y:Z):-1
//   */
//   deduce_and_test(fpir, 3, {FItv(1.0, 3.0), FItv(1.0, 1.0), FItv(0.0, 1.0), FItv(3.0, 3.0), FItv(0.0, 1.0), FItv(2.0,3.0)}, false);

//   interpret_must_succeed<IKind::TELL, true, false>("constraint float_eq(x, y);", fpir, env);
//   deduce_and_test(fpir, 4, {FItv(1.0, 3.0), FItv(1.0, 1.0), FItv(0.0, 1.0), FItv(3.0, 3.0), FItv(0.0, 1.0), FItv(2.0, 3.0)}, {FItv(3.0,3.0), FItv(1.0, 1.0), FItv(0.0, 0.0), FItv(3.0, 3.0), FItv(1.0, 1.0), FItv(3.0,3.0)}, true);
// }

// min(x, y) = z
TEST(FPIRTest, MinConstraint1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 0.0..4.0: x; var 2.0..5.0: y; var 0.0..10.0: z;\
    constraint float_min(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 10.0)}, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 4.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(z, 3.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 3.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(x, 1.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(2.0, 5.0), FItv(0.0, 3.0)}, {FItv(0.0, 1.0), FItv(2.0, 5.0), FItv(0.0, 1.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(x, 0.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 0.0), FItv(2.0, 5.0), FItv(0.0, 1.0)}, {FItv(0.0, 0.0), FItv(2.0, 5.0), FItv(0.0, 0.0)}, true);
}

// min(x, y) = z
TEST(FPIRTest, MinConstraint2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 0.0..4.0: x; var 2.0..5.0: y; var 0.0..10.0: z;\
    constraint float_min(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 10.0)}, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 4.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(z, 3.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 3.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(x, 4.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(4.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 3.0)}, {FItv(4.0, 4.0), FItv(2.0, 3.0), FItv(2.0, 3.0)}, false);
}

// min(x, y) = z
TEST(FPIRTest, MinConstraint3) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var 0.0..1.0: b1; var 0.0..1.0: b2;\
    constraint float_min(b1, b2, 1.0);", env);
  deduce_and_test(fpir, 1, {FItv::top(), FItv(0.0, 1.0), FItv(0.0, 1.0)}, {FItv::top(), FItv(1.0,1.0), FItv(1.0, 1.0)}, true);

  interpret_must_succeed<IKind::TELL, true, false>("constraint float_eq(b1, float_le(x, 5.0));", fpir, env);
  deduce_and_test(fpir, 2, {FItv::top(), FItv(1.0,1.0), FItv(1.0, 1.0)}, {FItv(FItv::LB::top(), 5.0), FItv(1.0, 1.0), FItv(1.0, 1.0)}, true);

  interpret_must_succeed<IKind::TELL, true, false>("constraint float_eq(b2, float_ge(x, 5.0));", fpir, env);
  deduce_and_test(fpir, 3, {FItv(FItv::LB::top(), 5.0), FItv(1.0, 1.0), FItv(1.0, 1.0)}, {FItv(5.0, 5.0), FItv(1.0, 1.0), FItv(1.0, 1.0)}, true);
}

// max(x, y) = z
TEST(FPIRTest, MaxConstraint1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 0.0..4.0: x; var 2.0..5.0: y; var 0.0..10.0: z;\
    constraint float_max(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 10.0)}, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(2.0, 5.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(z, 3.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(2.0, 3.0)}, {FItv(0.0, 3.0), FItv(2.0, 3.0), FItv(2.0, 3.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_le(x, 1.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(2.0, 3.0), FItv(2.0, 3.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(y, 2.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(2.0, 2.0), FItv(2.0, 3.0)}, {FItv(0.0, 1.0), FItv(2.0, 2.0), FItv(2.0, 2.0)}, true);
}

// max(x, y) = z
TEST(FPIRTest, MaxConstraint2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var 0.0..4.0: x; var 2.0..5.0: y; var 0.0..10.0: z;\
    constraint float_max(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(0.0, 10.0)}, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(2.0, 5.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_ge(z, 5.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 4.0), FItv(2.0, 5.0), FItv(5.0, 5.0)}, {FItv(0.0, 4.0), FItv(5.0, 5.0), FItv(5.0, 5.0)}, true);
}

TEST(FPIRTest, MaxConstraint3) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("var float: x; var 0.0..1.0: b1; var 0.0..1.0: b2;\
    constraint float_max(b1, b2, 0.0);", env);
  deduce_and_test(fpir, 1, {FItv::top(), FItv(0.0, 1.0), FItv(0.0, 1.0)}, {FItv::top(), FItv(0.0,0.0), FItv(0.0,0.0)}, true);

  interpret_must_succeed<IKind::TELL, true, false>("constraint float_eq(b1, float_le(x, 5.0));", fpir, env);
  deduce_and_test(fpir, 2, {FItv::top(), FItv(0.0,0.0), FItv(0.0,0.0)}, {FItv::top(), FItv(0.0, 0.0), FItv(0.0, 0.0)}, false);

  interpret_must_succeed<IKind::TELL, true, false>("constraint float_eq(b2, float_ge(x, 7.0));", fpir, env);
  deduce_and_test(fpir, 3, {FItv::top(), FItv(0.0, 0.0), FItv(0.0, 0.0)}, {FItv::top(), FItv(0.0, 0.0), FItv(0.0, 0.0)}, false);
}

TEST(FPIRTest, FloatTimes1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..1.0: x;\
    var 0.0..1.0: y;\
    var 0.0..1.0: z;\
    constraint float_times(x,y,z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(0.0, 1.0), FItv(0.0, 1.0)}, false);
  interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(x, 1.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(1.0, 1.0), FItv(0.0, 1.0), FItv(0.0, 1.0)}, false);
  // interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(y, 1.0);", fpir, env);
  // deduce_and_test(fpir, 1, {FItv(1.0, 1.0), FItv(1.0, 1.0), FItv(0.0, 1.0)}, {FItv(1.0, 1.0), FItv(1.0, 1.0), FItv(1.0, 1.0)}, true);
}

TEST(FPIRTest, FloatTimes2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..1.0: x;\
    var 0.0..1.0: y;\
    var 0.0..1.0: z;\
    constraint float_times(x,y,z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(0.0, 1.0), FItv(0.0, 1.0)}, false);
  interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(z, 1.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(0.0, 1.0), FItv(1.0, 1.0)}, {FItv(1.0, 1.0), FItv(1.0, 1.0), FItv(1.0, 1.0)}, true);
}

TEST(FPIRTest, FloatTimes3) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..0.0: x; \
    var 0.0..1.0: y; \
    var 0.0..1.0: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 0.0), FItv(0.0, 1.0), FItv(0.0, 1.0)}, {FItv(0.0, 0.0), FItv(0.0, 1.0), FItv(0.0, 0.0)}, true);
}

TEST(FPIRTest, FloatTimes4) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..1.0: x; \
    var 0.0..0.0: y; \
    var 0.0..1.0: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(0.0, 0.0), FItv(0.0, 1.0)}, {FItv(0.0, 1.0), FItv(0.0, 0.0), FItv(0.0, 0.0)}, true);
}

TEST(FPIRTest, FloatTimes5) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 1.0..2.0: x; \
    var 0.0..1.0: y; \
    var 0.0..0.0: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(1.0, 2.0), FItv(0.0, 1.0), FItv(0.0, 0.0)}, {FItv(1.0, 2.0), FItv(0.0, 0.0), FItv(0.0, 0.0)}, true);
}

TEST(FPIRTest, FloatTimes6) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..1.0: x; \
    var 1.0..2.0: y; \
    var 0.0..0.0: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test(fpir, 1, {FItv(0.0, 1.0), FItv(1.0, 2.0), FItv(0.0, 0.0)}, {FItv(0.0, 0.0), FItv(1.0, 2.0), FItv(0.0, 0.0)}, true);
}

TEST(FPIRTest, FloatTimes7) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..1.0: x; \
    var 0.99999..5.123456: y; \
    var float: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test(fpir, 1, 
    {create_float_interval<FItv>("0.0", "1.0"), create_float_interval<FItv>("0.99999", "5.123456"), FItv::top()}, 
    {create_float_interval<FItv>("0.0", "1.0"), create_float_interval<FItv>("0.99999", "5.123456"), FItv(0.0, 5.123456)}, 
    false);
}

TEST(FPIRTest, FloatTimes8) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 10.3322111..11.9877777: x; \
    var 0.99999..5.123456: y; \
    var float: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test(fpir, 1, 
    {create_float_interval<FItv>("10.3322111", "11.9877777"), create_float_interval<FItv>("0.99999", "5.123456"), FItv::top()}, 
    {create_float_interval<FItv>("10.3322111", "11.9877777"), create_float_interval<FItv>("0.99999", "5.123456"), FItv(10.332107777888996, 61.418851583731203)}, 
    false);
}

TEST(FPIRTest, FloatTimes9) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var 0.0..1.0: x; \
    var 0.99999..5.123456: y; \
    var 10.3322111..11.9877777: z; \
    constraint float_times(x, y, z);", env);
  deduce_and_test_bot(fpir, 1, {create_float_interval<FItv>("0.0", "1.0"), create_float_interval<FItv>("0.99999", "5.123456"), create_float_interval<FItv>("10.3322111", "11.9877777")});
}

TEST(FPIRTest, FloatAbs1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var -15.0..5.0: x;\
    var -10.0..10.0: y;\
    constraint float_abs(x, y);", env);
  deduce_and_test(fpir, 3, {FItv(-15.0, 5.0), FItv(-10.0, 10.0)}, {FItv(-10.0, 5.0), FItv(0.0, 10.0)}, false);
}

TEST(FPIRTest, InfiniteDomain1) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var float: x; \
    var float: b; \
    constraint float_ge(b, 0.0); \
    constraint float_le(b, 1.0); \
    constraint float_eq(b, float_le(x,5.0));", env);
  deduce_and_test(fpir, 1, {FItv::top(), FItv(0.0,1.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(b, 1.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv::top(), FItv(1.0,1.0)}, {FItv(FItv::LB::top(), 5.0), FItv(1.0,1.0)}, true);
}

TEST(FPIRTest, InfiniteDomain2) {
  VarEnv<standard_allocator> env;
  FPIR fpir = create_and_interpret_and_tell<FPIR, true, false>("\
    var float: x; \
    var float: b; \
    constraint float_ge(b, 0.0); \
    constraint float_le(b, 1.0); \
    constraint float_eq(b, float_le(x,5.0));", env);
  deduce_and_test(fpir, 1, {FItv::top(), FItv(0.0,1.0)}, false);

  interpret_must_succeed<IKind::TELL, false, false>("constraint float_eq(b, 0.0);", fpir, env);
  deduce_and_test(fpir, 1, {FItv::top(), FItv(0.0,0.0)}, {FItv::top(), FItv(0.0,0.0)}, false);
}