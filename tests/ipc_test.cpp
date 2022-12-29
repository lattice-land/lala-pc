// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include "abstract_testing.hpp"

#include "vstore.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "ipc.hpp"
#include "terms.hpp"
#include "fixpoint.hpp"
#include "vector.hpp"
#include "shared_ptr.hpp"

using namespace lala;
using namespace battery;

using F = TFormula<StandardAllocator>;

using zi = local::ZInc;
using zd = local::ZDec;
using Itv = Interval<zi>;
using IStore = VStore<Itv, StandardAllocator>;
using IIPC = IPC<IStore>;

const AType sty = 0;
const AType pty = 1;

TEST(TermTest, AddTermBinary) {
  IStore store = interpret_to2<IStore>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);");
  Add<Variable<IStore>, Variable<IStore>> x_plus_y(
    Variable<IStore>(AVar(sty, 0)),
    Variable<IStore>(AVar(sty, 1)));
  EXPECT_EQ(x_plus_y.project(store), Itv(0,20));
  local::BInc has_changed2;
  x_plus_y.tell(store, Itv(zi::bot(), 5), has_changed2);
  EXPECT_TRUE(has_changed2);
  EXPECT_EQ(x_plus_y.project(store), Itv(0,10));
}

TEST(TermTest, AddTermNary) {
  IStore store = interpret_to2<IStore>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_ge(z, 0); constraint int_le(z, 10);");
  vector<Variable<IStore>, StandardAllocator> vars(3);
  for(int i = 0; i < 3; ++i) {
    vars[i] = Variable<IStore>(AVar(sty, i));
  }
  NaryAdd<Variable<IStore>, StandardAllocator> sum_xyz(std::move(vars));
  EXPECT_EQ(sum_xyz.project(store), Itv(0,30));
  local::BInc has_changed2;
  sum_xyz.tell(store, Itv(zi::bot(), 5), has_changed2);
  EXPECT_TRUE(has_changed2);
  EXPECT_EQ(sum_xyz.project(store), Itv(0,15));
}

template <class L>
void test_extract(const L& ipc, bool is_ua) {
  AbstractDeps<StandardAllocator> deps;
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
  local::BInc has_changed = GaussSeidelIteration::fixpoint(ipc);
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
  IIPC ipc = interpret_to2<IIPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_plus(x, y, 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(0,5)}, false);
}

// x + y = z, z <= 5
TEST(IPCTest, TemporalConstraint1Flat) {
  IIPC ipc = interpret_to2<IIPC>("var int: x; var int: y; var int: z;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(z, 5);\
    constraint int_plus(x, y, z);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10), Itv(zi::bot(), zd(5))}, {Itv(0,5), Itv(0,5), Itv(0,5)}, false);
}

TEST(IPCTest, TemporalConstraint1) {
  IIPC ipc = interpret_to2<IIPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_le(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, {Itv(0,5), Itv(0,5)}, false);
}

// x + y > 5 (x,y in [0..10])
TEST(IPCTest, TemporalConstraint2) {
  IIPC ipc = interpret_to2<IIPC>("var int: x; var int: y;\
    constraint int_ge(x, 0); constraint int_le(x, 10);\
    constraint int_ge(y, 0); constraint int_le(y, 10);\
    constraint int_gt(int_plus(x, y), 5);");
  refine_and_test(ipc, 1, {Itv(0,10), Itv(0,10)}, false);
}

// // x + y > 5 (x,y in [0..3])
// TEST(IPCTest, TemporalConstraint3) {
//   IIPC ipc = binary_op(ADD, 0, 3, GT, 5, true);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(3, 3));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(3, 3));
//   test_underapproximation(ipc, true);
// }

// // x + y >= 5 (x,y in [0..3])
// TEST(IPCTest, TemporalConstraint4) {
//   IIPC ipc = binary_op(ADD, 0, 3, GEQ, 5, true);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(2, 3));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(2, 3));
// }

// // x + y = 5 (x,y in [0..4])
// TEST(IPCTest, TemporalConstraint5) {
//   IIPC ipc = binary_op(ADD, 0, 4, EQ, 5, true);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(1, 4));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(1, 4));
//   test_underapproximation(ipc, false);
// }

// // x - y <= 5 (x,y in [0..10])
// TEST(IPCTest, TemporalConstraint6) {
//   IIPC ipc = binary_op(SUB, 0, 10, LEQ, 5, false);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 10));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 10));
//   test_underapproximation(ipc, false);
// }

// // x - y <= -10 (x,y in [0..10])
// TEST(IPCTest, TemporalConstraint7) {
//   IIPC ipc = binary_op(SUB, 0, 10, LEQ, -10, true);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 0));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(10, 10));
//   test_underapproximation(ipc, true);
// }

// // x - y >= 5 (x,y in [0..10])
// TEST(IPCTest, TemporalConstraint8) {
//   IIPC ipc = binary_op(SUB, 0, 10, GEQ, 5, true);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(5, 10));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 5));
//   test_underapproximation(ipc, false);
// }

// // x - y <= -5 (x,y in [0..10])
// TEST(IPCTest, TemporalConstraint9) {
//   IIPC ipc = binary_op(SUB, 0, 10, LEQ, -5, true);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 5));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(5, 10));
//   test_underapproximation(ipc, false);
// }

// // x <= -5 + y (x,y in [0..10])
// TEST(IPCTest, TemporalConstraint10) {
//   int lb = 0;
//   int ub = 10;
//   IIPC ipc(pty, make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty))));
//   F::Sequence doms(6);
//   doms[0] = F::make_exists(sty, var_x, Int);
//   doms[1] = F::make_exists(sty, var_y, Int);
//   doms[2] = make_v_op_z(var_x, LEQ, ub);
//   doms[3] = make_v_op_z(var_y, LEQ, ub);
//   doms[4] = make_v_op_z(var_x, GEQ, lb);
//   doms[5] = make_v_op_z(var_y, GEQ, lb);
//   F::Sequence all(2);
//   all[0] = F::make_nary(AND, doms, sty);
//   all[1] = F::make_binary(
//     F::make_lvar(sty, var_x),
//     LEQ,
//     F::make_binary(
//       F::make_z(-5),
//       ADD,
//       F::make_lvar(sty, var_y), pty),
//     pty);
//   auto f = F::make_nary(AND, all, pty);
//   thrust::optional<IIPC::TellType> res(ipc.interpret(f));
//   EXPECT_TRUE(res.has_value());
//   BInc has_changed = BInc::bot();
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed.is_top());
//   EXPECT_EQ(ipc.num_refinements(), 1);

//   // Propagation tests
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(lb, ub));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(lb, ub));

//   BInc has_changed2 = BInc::bot();
//   ipc.refine(has_changed2);
//   EXPECT_EQ(has_changed2.is_top(), true);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 5));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(5, 10));
// }


// template <class A>
// void vars3_of(const A& a, AVar vars[3]) {
//   vars[0] = *(a.environment().to_avar(var_x));
//   vars[1] = *(a.environment().to_avar(var_y));
//   vars[2] = *(a.environment().to_avar(var_z));
// }

// IIPC ternary_op(Sig sig, int lb, int ub, int k) {
//   IIPC ipc(pty, make_shared<IStore, StandardAllocator>(std::move(xyz_store(lb, ub))));
//   F::Sequence terms(3);
//   EXPECT_EQ(terms.size(), 3);
//   terms[0] = F::make_lvar(sty, var_x);
//   terms[1] = F::make_lvar(sty, var_y);
//   terms[2] = F::make_lvar(sty, var_z);
//   auto sum = F::make_nary(sig, std::move(terms), pty);
//   auto cons = F::make_binary(sum, LEQ, F::make_z(k), pty);
//   thrust::optional<IIPC::TellType> res(ipc.interpret(cons));
//   EXPECT_TRUE(res.has_value());
//   BInc has_changed = BInc::bot();
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed.is_top());
//   EXPECT_EQ(ipc.num_refinements(), 1);
//   EXPECT_EQ(ipc.environment().size(), 3);

//   // Propagation tests
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(lb, ub));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(lb, ub));
//   EXPECT_EQ(ipc.project(vars[2]), Itv(lb, ub));

//   BInc has_changed2 = BInc::bot();
//   ipc.refine(has_changed2);
//   EXPECT_TRUE(has_changed2.is_top());
//   return std::move(ipc);
// }

// // TOP test x,y,z in [3..10] /\ x + y + z <= 8
// TEST(IPCTest, TopProp) {
//   IIPC ipc = ternary_op(ADD, 3, 10, 8);
//   EXPECT_EQ(ipc.environment().size(), 3);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_TRUE(ipc.is_top());
//   EXPECT_TRUE(ipc.project(vars[0]).is_top());
//   EXPECT_TRUE(ipc.project(vars[1]).is_top());
//   EXPECT_TRUE(ipc.project(vars[2]).is_top());
//   test_underapproximation(ipc, false);
// }

// // x,y,z in [3..10] /\ x + y + z <= 9
// TEST(IPCTest, TernaryAdd2) {
//   IIPC ipc = ternary_op(ADD, 3, 10, 9);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   for(int i = 0; i < 3; ++i) {
//     EXPECT_EQ(ipc.project(vars[i]), Itv(3, 3));
//   }
//   test_underapproximation(ipc, true);
// }

// // x,y,z in [3..10] /\ x + y + z <= 10
// TEST(IPCTest, TernaryAdd3) {
//   IIPC ipc = ternary_op(ADD, 3, 10, 10);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   for(int i = 0; i < 3; ++i) {
//     EXPECT_EQ(ipc.project(vars[i]), Itv(3, 4));
//   }
//   test_underapproximation(ipc, false);
// }

// // x,y,z in [-2..2] /\ x + y + z <= -5
// TEST(IPCTest, TernaryAdd4) {
//   IIPC ipc = ternary_op(ADD, -2, 2, -5);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   for(int i = 0; i < 3; ++i) {
//     EXPECT_EQ(ipc.project(vars[i]), Itv(-2, -1));
//   }
//   test_underapproximation(ipc, false);
// }

// // Constraint of the form k1 * x + k2 * y + k3 * z <= k, with x,y,z in [0..1].
// IIPC pseudo_boolean(int k1, int k2, int k3, int k, bool expect_changed) {
//   IIPC ipc(pty, make_shared<IStore, StandardAllocator>(std::move(xyz_store(0,1))));
//   F::Sequence terms(3);
//   EXPECT_EQ(terms.size(), 3);
//   terms[0] = F::make_binary(F::make_z(k1), MUL, F::make_lvar(sty, var_x));
//   terms[1] = F::make_binary(F::make_z(k2), MUL, F::make_lvar(sty, var_y));
//   terms[2] = F::make_binary(F::make_z(k3), MUL, F::make_lvar(sty, var_z));
//   auto sum = F::make_nary(ADD, std::move(terms), pty);
//   auto cons = F::make_binary(sum, LEQ, F::make_z(k), pty);
//   thrust::optional<IIPC::TellType> res(ipc.interpret(cons));
//   EXPECT_TRUE(res.has_value());
//   BInc has_changed = BInc::bot();
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed.is_top());
//   EXPECT_EQ(ipc.num_refinements(), 1);
//   EXPECT_EQ(ipc.environment().size(), 3);

//   // Propagation tests
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[2]), Itv(0, 1));

//   BInc has_changed2 = BInc::bot();
//   ipc.refine(has_changed2);
//   EXPECT_EQ(expect_changed, has_changed2.is_top());
//   return std::move(ipc);
// }

// // x,y,z in [0..1] /\ 2x + y + 3z <= 2
// TEST(IPCTest, PseudoBoolean1) {
//   IIPC ipc = pseudo_boolean(2, 1, 3, 2, true);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[2]), Itv(0, 0));
//   test_underapproximation(ipc, false);
// }

// // x,y,z in [0..1] /\ 2x + 5y + 3z <= 2
// TEST(IPCTest, PseudoBoolean2) {
//   IIPC ipc = pseudo_boolean(2, 5, 3, 2, true);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 0));
//   EXPECT_EQ(ipc.project(vars[2]), Itv(0, 0));
//   test_underapproximation(ipc, true);
// }

// // x,y,z in [0..1] /\ 3x + 5y + 3z <= 2
// TEST(IPCTest, PseudoBoolean3) {
//   IIPC ipc = pseudo_boolean(3, 5, 3, 2, true);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 0));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 0));
//   EXPECT_EQ(ipc.project(vars[2]), Itv(0, 0));
//   test_underapproximation(ipc, true);
// }

// // x,y,z in [0..1] /\ -x + y + 3z <= 2
// TEST(IPCTest, PseudoBoolean4) {
//   IIPC ipc = pseudo_boolean(-1, 1, 3, 2, false);
//   AVar vars[3];
//   vars3_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(0, 1));
//   EXPECT_EQ(ipc.project(vars[2]), Itv(0, 1));
//   test_underapproximation(ipc, false);
// }

// // Constraint of the form <unop> x <op> k.
// IIPC unop_constraint(Sig unop, int lb, int ub, Sig binop, int k, bool expect_changed) {
//   IIPC ipc(pty, make_shared<IStore, StandardAllocator>(std::move(x_store(lb, ub))));
//   auto cons = F::make_binary(
//     F::make_unary(unop, F::make_lvar(sty, var_x), pty),
//     binop,
//     F::make_z(k), pty);
//   thrust::optional<IIPC::TellType> res(ipc.interpret(cons));
//   EXPECT_TRUE(res.has_value());
//   BInc has_changed = BInc::bot();
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed.is_top());
//   EXPECT_EQ(ipc.num_refinements(), 1);
//   EXPECT_EQ(ipc.environment().size(), 1);

//   // Propagation tests
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(lb, ub));

//   BInc has_changed2 = BInc::bot();
//   ipc.refine(has_changed2);
//   EXPECT_EQ(expect_changed, has_changed2.is_top());
//   return std::move(ipc);
// }

// // x in [-4..3], -x <= 2
// TEST(IPCTest, NegationOp1) {
//   IIPC ipc = unop_constraint(NEG, -4, 3, LEQ, 2, true);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(-2, 3));
//   test_underapproximation(ipc, true);
// }

// // x in [-4..3], -x <= -2
// TEST(IPCTest, NegationOp2) {
//   IIPC ipc = unop_constraint(NEG, -4, 3, LEQ, -2, true);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(2, 3));
//   test_underapproximation(ipc, true);
// }

// // x in [0..3], -x <= -2
// TEST(IPCTest, NegationOp3) {
//   IIPC ipc = unop_constraint(NEG, 0, 3, LEQ, -2, true);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(2, 3));
//   test_underapproximation(ipc, true);
// }

// // x in [-4..-3], -x <= 4
// TEST(IPCTest, NegationOp4) {
//   IIPC ipc = unop_constraint(NEG, -4, -3, LEQ, 4, false);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(-4, -3));
//   test_underapproximation(ipc, true);
// }

// // x in [-4..3], -x >= -2
// TEST(IPCTest, NegationOp5) {
//   IIPC ipc = unop_constraint(NEG, -4, 3, GEQ, -2, true);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(-4, 2));
//   test_underapproximation(ipc, true);
// }

// // x in [-4..3], -x > 2
// TEST(IPCTest, NegationOp6) {
//   IIPC ipc = unop_constraint(NEG, -4, 3, GT, 2, true);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_EQ(ipc.project(var), Itv(-4, -3));
//   test_underapproximation(ipc, true);
// }

// // x in [-4..3], -x >= 5
// TEST(IPCTest, NegationOp7) {
//   IIPC ipc = unop_constraint(NEG, -4, 3, GEQ, 5, true);
//   AVar var = *(ipc.environment().to_avar(var_x));
//   EXPECT_TRUE(ipc.project(var).is_top());
//   test_underapproximation(ipc, false);
// }

// // Create a constraint of the form "b <=> (x - y <= k1 /\ y - x <= k2)".
// IIPC make_resource_cons(int lbx, int ubx, int lby, int uby, int k1, int k2) {
//   IIPC ipc(pty, make_shared<IStore, StandardAllocator>(std::move(IStore::bot(sty))));
//   F::Sequence doms(9);
//   doms[0] = F::make_exists(sty, var_x, Int);
//   doms[1] = F::make_exists(sty, var_y, Int);
//   doms[2] = make_v_op_z(var_x, LEQ, ubx);
//   doms[3] = make_v_op_z(var_y, LEQ, uby);
//   doms[4] = make_v_op_z(var_x, GEQ, lbx);
//   doms[5] = make_v_op_z(var_y, GEQ, lby);
//   doms[6] = F::make_exists(sty, var_b, Int);
//   doms[7] = make_v_op_z(var_b, LEQ, 1);
//   doms[8] = make_v_op_z(var_b, GEQ, 0);

//   F::Sequence right(2);
//   right[0] = F::make_binary(F::make_binary(F::make_lvar(sty, var_x), SUB, F::make_lvar(sty, var_y), pty), LEQ, F::make_z(k1), pty);
//   right[1] = F::make_binary(F::make_binary(F::make_lvar(sty, var_y), SUB, F::make_lvar(sty, var_x), pty), LEQ, F::make_z(k2), pty);

//   F::Sequence all(2);
//   all[0] = F::make_nary(AND, doms, sty);
//   all[1] = F::make_binary(F::make_lvar(pty, var_b), EQUIV,
//     F::make_nary(AND, right, pty), pty);
//   auto f = F::make_nary(AND, all, pty);

//   thrust::optional<IIPC::TellType> res(ipc.interpret(f));
//   EXPECT_TRUE(res.has_value());
//   BInc has_changed = BInc::bot();
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed.is_top());
//   EXPECT_EQ(ipc.num_refinements(), 1);

//   // Projection tests
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(lbx, ubx));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(lby, uby));
//   return std::move(ipc);
// }

// TEST(IPCTest, ResourceConstraint1) {
//   IIPC ipc = make_resource_cons(5, 10, 9, 15, 0, 2);

//   BInc has_changed = BInc::bot();
//   ipc.refine(has_changed);
//   EXPECT_FALSE(has_changed.is_top());

//   // Add b = true in the abstract element
//   thrust::optional<IIPC::TellType> res(ipc.interpret(F::make_binary(F::make_lvar(sty, var_b), EQ, F::make_z(1), sty)));
//   EXPECT_TRUE(res.has_value());
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed);
//   EXPECT_EQ(ipc.num_refinements(), 1);
//   AVar b = *(ipc.environment().to_avar(var_b));
//   EXPECT_EQ(ipc.project(b), Itv(1, 1));

//   // Run the refinement operator.
//   has_changed.dtell(BInc::bot());
//   ipc.refine(has_changed);
//   EXPECT_TRUE(has_changed);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(7, 10));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(9, 12));
//   test_underapproximation(ipc, false);
// }

// TEST(IPCTest, ResourceConstraint2) {
//   IIPC ipc = make_resource_cons(1, 2, 0, 2, 2, -1);

//   BInc has_changed = BInc::bot();
//   ipc.refine(has_changed);
//   EXPECT_FALSE(has_changed.is_top());

//   // Add b = false in the abstract element
//   thrust::optional<IIPC::TellType> res(ipc.interpret(F::make_binary(F::make_lvar(sty, var_b), EQ, F::make_z(0), sty)));
//   EXPECT_TRUE(res.has_value());
//   ipc.tell(std::move(*res), has_changed);
//   EXPECT_TRUE(has_changed);
//   EXPECT_EQ(ipc.num_refinements(), 1);
//   AVar b = *(ipc.environment().to_avar(var_b));
//   EXPECT_EQ(ipc.project(b), Itv(0, 0));

//   // Run the refinement operator.
//   has_changed.dtell(BInc::bot());
//   ipc.refine(has_changed);
//   EXPECT_TRUE(has_changed);
//   AVar vars[2];
//   vars2_of(ipc, vars);
//   EXPECT_EQ(ipc.project(vars[0]), Itv(1, 2));
//   EXPECT_EQ(ipc.project(vars[1]), Itv(1, 2));
//   test_underapproximation(ipc, false);
// }

// TEST(IPCTest, CloneIPC) {
//   IIPC ipc = binary_op(ADD, 0, 10, LEQ, 5, true);
//   AbstractDeps<StandardAllocator> deps;
//   IIPC copy1(ipc, deps);
//   EXPECT_EQ(deps.size(), 1);
//   AVar vars[2];
//   vars2_of(copy1, vars);
//   EXPECT_EQ(copy1.project(vars[0]), Itv(0, 5));
//   EXPECT_EQ(copy1.project(vars[1]), Itv(0, 5));
//   IIPC copy2(ipc, deps);
//   EXPECT_EQ(deps.size(), 1);
//   vars2_of(copy2, vars);
//   EXPECT_EQ(copy2.project(vars[0]), Itv(0, 5));
//   EXPECT_EQ(copy2.project(vars[1]), Itv(0, 5));
// }
