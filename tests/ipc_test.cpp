// Copyright 2021 Pierre Talbot

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "z.hpp"
#include "cartesian_product.hpp"
#include "interval.hpp"
#include "vstore.hpp"
#include "ipc.hpp"
#include "terms.hpp"

using namespace lala;

using F = TFormula<StandardAllocator>;

static LVar<StandardAllocator> var_x = "x";
static LVar<StandardAllocator> var_y = "y";

#define EXPECT_EQ2(a,b) EXPECT_EQ(unwrap(a), unwrap(b))
#define EXPECT_TRUE2(a) EXPECT_TRUE(unwrap(a))
#define EXPECT_FALSE2(a) EXPECT_FALSE(unwrap(a))

using zi = ZInc<int>;
using Itv = Interval<zi>;
using IStore = VStore<Itv, StandardAllocator>;
using IIPC = IPC<IStore>;

const AType sty = 0;
const AType pty = 1;

TEST(TermTest, AddTerm) {
  IStore store = IStore::bot(sty);
  F::Sequence doms(6);
  doms[0] = F::make_exists(sty, var_x, Int);
  doms[1] = F::make_exists(sty, var_y, Int);
  doms[2] = make_v_op_z(var_x, LEQ, 10);
  doms[3] = make_v_op_z(var_y, LEQ, 10);
  doms[4] = make_v_op_z(var_x, GEQ, 0);
  doms[5] = make_v_op_z(var_y, GEQ, 0);
  auto f = F::make_nary(AND, doms, sty);
  auto cons = store.interpret(f);
  BInc has_changed = BInc::bot();
  store.tell(*cons, has_changed);
  DArray<Variable<IStore>, StandardAllocator> vars(2);
  vars[0] = Variable<IStore>(make_var(sty, 0));
  vars[1] = Variable<IStore>(make_var(sty, 1));
  NaryAdd<Variable<IStore>, StandardAllocator> x_plus_y(std::move(vars));
  EXPECT_EQ2(x_plus_y.project(store), Itv(0,20));
  BInc has_changed2 = BInc::bot();
  x_plus_y.tell(store, Itv(zi::bot(), 5), has_changed2);
  EXPECT_TRUE2(has_changed2);
  EXPECT_EQ2(x_plus_y.project(store), Itv(0,10));
}

TEST(TermTest, AddTermBinary) {
  IStore store = IStore::bot(sty);
  F::Sequence doms(6);
  doms[0] = F::make_exists(sty, var_x, Int);
  doms[1] = F::make_exists(sty, var_y, Int);
  doms[2] = make_v_op_z(var_x, LEQ, 10);
  doms[3] = make_v_op_z(var_y, LEQ, 10);
  doms[4] = make_v_op_z(var_x, GEQ, 0);
  doms[5] = make_v_op_z(var_y, GEQ, 0);
  auto f = F::make_nary(AND, doms, sty);
  auto cons = store.interpret(f);
  BInc has_changed = BInc::bot();
  store.tell(*cons, has_changed);

  Add<Variable<IStore>, Variable<IStore>> x_plus_y(
    Variable<IStore>(make_var(sty, 0)),
    Variable<IStore>(make_var(sty, 1)));
  EXPECT_EQ2(x_plus_y.project(store), Itv(0,20));
  BInc has_changed2 = BInc::bot();
  x_plus_y.tell(store, Itv(zi::bot(), 5), has_changed2);
  EXPECT_TRUE2(has_changed2);
  EXPECT_EQ2(x_plus_y.project(store), Itv(0,10));
}

// x + y <= 5
TEST(IPCTest, TemporalConstraint) {
  IStore* istore = new IStore(IStore::bot(sty));
  IIPC ipc(pty, istore);
  F::Sequence doms(6);
  doms[0] = F::make_exists(sty, var_x, Int);
  doms[1] = F::make_exists(sty, var_y, Int);
  doms[2] = make_v_op_z(var_x, LEQ, 10);
  doms[3] = make_v_op_z(var_y, LEQ, 10);
  doms[4] = make_v_op_z(var_x, GEQ, 0);
  doms[5] = make_v_op_z(var_y, GEQ, 0);
  F::Sequence all(2);
  all[0] = F::make_nary(AND, doms, sty);
  all[1] = F::make_binary(F::make_binary(F::make_lvar(sty, var_x), ADD, F::make_lvar(sty, var_y), pty), LEQ, F::make_z(5), pty);
  auto f = F::make_nary(AND, all, pty);
  thrust::optional<IIPC::TellType> res(ipc.interpret(f));
  EXPECT_TRUE(res.has_value());
  BInc has_changed = BInc::bot();
  ipc.tell(std::move(*res), has_changed);
  EXPECT_TRUE2(has_changed.is_top());
  EXPECT_EQ(ipc.num_refinements(), 1);

  // Propagation tests
  auto x = *(ipc.environment().to_avar(var_x));
  auto y = *(ipc.environment().to_avar(var_y));
  EXPECT_EQ2(ipc.project(x), Itv(0, 10));
  EXPECT_EQ2(ipc.project(y), Itv(0, 10));

  BInc has_changed2 = BInc::bot();
  ipc.refine(has_changed2);
  EXPECT_TRUE2(has_changed.is_top());
  EXPECT_EQ2(ipc.project(x), Itv(0, 5));
  EXPECT_EQ2(ipc.project(y), Itv(0, 5));
}

