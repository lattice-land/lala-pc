// Copyright 2025 Pierre Talbot

#include "abstract_testing.hpp"
#include "battery/vector.hpp"
#include "lala/vstore.hpp"
#include "lala/cartesian_product.hpp"
#include "lala/interval.hpp"
#include "lala/pir.hpp"
#include "lala/terms.hpp"
#include "lala/fixpoint.hpp"
#include "abstract_testing.hpp"

#include <format>

using zlb = local::ZLB;
using zub = local::ZUB;
using Itv = Interval<zlb>;

using flb = local::FLB; 
using fub = local::FUB;
using FItv = Interval<flb>;

template<class L>
void deduce_and_test2(L& ipc, const std::vector<Itv>& before, const std::vector<Itv>& after, bool test_completeness, bool disable_before_test) {
  bool has_bot = std::ranges::any_of(after, [](const Itv& itv) { return itv.is_bot(); });
  // In case bottom is discovered before starting the fixpoint iteration.
  if(ipc.is_bot()) {
    EXPECT_TRUE(has_bot);
    return;
  }
  if(!disable_before_test) {
    for(int i = 0; i < before.size(); ++i) {
      EXPECT_EQ(ipc[i], before[i]) << "ipc[" << i << "]";
    }
  }
  // If all intervals are singleton from the beginning, we must test for completeness.
  if(std::ranges::all_of(before, [](const Itv& itv) { return itv.lb().value() == itv.ub().value(); })) {
    test_completeness = true;
  }
  local::B has_changed = false;
  GaussSeidelIteration{}.fixpoint(
    ipc.num_deductions(),
    [&](size_t i) { return ipc.deduce(i); },
    has_changed);
  if(ipc.is_bot()) {
    EXPECT_TRUE(has_bot);
  }
  else {
    for(int i = 0; i < after.size(); ++i) {
      if(test_completeness) {
        EXPECT_EQ(ipc[i], after[i]) << ipc[i] << " == " << after[i] << "ipc[" << i << "]";
      }
      else {
        EXPECT_TRUE(ipc[i] >= after[i]) << ipc[i] << " >= " << after[i] << " is false (unsound) ipc[" << i << "]";
      }
    }
  }
}

template<class L>
void fdeduce_and_test2(L& fpc, const std::vector<FItv>& before, const std::vector<FItv>& after, bool test_completeness, bool disable_before_test) {
  bool has_bot = std::ranges::any_of(after, [](const FItv& fitv) { return fitv.is_bot(); });
  // In case bottom is discovered before starting the fixpoint iteration.
  if(fpc.is_bot()) {
    EXPECT_TRUE(has_bot);
    return;
  }
  if(!disable_before_test) {
    for(int i = 0; i < before.size(); ++i) {
      EXPECT_EQ(fpc[i], before[i]) << "fpc[" << i << "]";
    }
  }
  // If all intervals are singleton from the beginning, we must test for completeness.
  if(std::ranges::all_of(before, [](const FItv& fitv) { return fitv.lb().value() == fitv.ub().value(); })) {
    test_completeness = true;
  }
  local::B has_changed = false;
  GaussSeidelIteration{}.fixpoint(
    fpc.num_deductions(),
    [&](size_t i) { return fpc.deduce(i, false); },
    has_changed);
  if(fpc.is_bot()) {
    EXPECT_TRUE(has_bot);
  }
  else {
    for(int i = 0; i < after.size(); ++i) {
      if(test_completeness) {
        EXPECT_EQ(fpc[i], after[i]) << fpc[i] << " == " << after[i] << "fpc[" << i << "]";
      }
      else {
        EXPECT_TRUE(fpc[i] >= after[i]) << fpc[i] << " >= " << after[i] << " is false (unsound) fpc[" << i << "]";
      }
    }
  }
}

template <class A, class F>
void test_bound_propagator_soundness(const char* pred_name, F pred, bool test_completeness = true, bool disable_before_test = false) {
  int minval = -20;
  int maxval = 20;
  std::vector<Itv> itvs{Itv(minval, minval), Itv(minval+1, maxval-1), Itv(minval,-3), Itv(minval,-2), Itv(minval,0), Itv(minval,maxval), Itv(-2,-1), Itv(-1,0), Itv(-2,2), Itv(-1,-1), Itv(0,0), Itv(1,1), Itv(2,2), Itv(0,1), Itv(1,2), Itv(0, maxval), Itv(2, maxval), Itv(3, maxval), Itv(maxval, maxval), Itv(1,0), Itv(2,10), Itv(-25,25), Itv(-2, 3)};
  for(int i = 0; i < itvs.size(); ++i) {
    for(int j = 0; j < itvs.size(); ++j) {
      for(int k = 0; k < itvs.size(); ++k) {
        Itv x = itvs[i];
        Itv y = itvs[j];
        Itv z = itvs[k];

        std::string fzn = std::format("var {}..{}: x; var {}..{}: y; var {}..{}: z;\
          constraint {}(y, z, x);", x.lb().value(), x.ub().value(), y.lb().value(), y.ub().value(), z.lb().value(), z.ub().value(), pred_name);

        std::vector<int> xs, ys, zs;
        for(int a = x.lb().value(); a <= x.ub().value(); ++a) {
          for(int b = y.lb().value(); b <= y.ub().value(); ++b) {
            for(int c = z.lb().value(); c <= z.ub().value(); ++c) {
              if(pred(a, b, c)) {
                xs.push_back(a);
                ys.push_back(b);
                zs.push_back(c);
              }
            }
          }
        }
        Itv x2 = xs.empty() ? Itv::bot() : Itv(std::ranges::min(xs), std::ranges::max(xs));
        Itv y2 = ys.empty() ? Itv::bot() : Itv(std::ranges::min(ys), std::ranges::max(ys));
        Itv z2 = zs.empty() ? Itv::bot() : Itv(std::ranges::min(zs), std::ranges::max(zs));

        A a = create_and_interpret_and_tell<A, true>(fzn.data());
        deduce_and_test2(a, {x,y,z}, {x2,y2,z2}, test_completeness, disable_before_test);

        if(!a.is_bot()) {
          bool abstract_entailed = true;
          for(int i = 0; i < a.num_deductions(); ++i) {
            abstract_entailed &= a.ask(i);
          }
          bool is_bot = x2.is_bot() || y2.is_bot() || z2.is_bot();
          if(is_bot) {
            EXPECT_FALSE(abstract_entailed);
          }
          else {
            if(abstract_entailed) {
              for(int a = x2.lb().value(); a <= x2.ub().value(); ++a) {
                for(int b = y2.lb().value(); b <= y2.ub().value(); ++b) {
                  for(int c = z2.lb().value(); c <= z2.ub().value(); ++c) {
                    EXPECT_TRUE(pred(a, b, c)) << "The constraint is not entailed on (" << a << ", " << b << ", " << c << ") " << fzn;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <class A, class F>
void test_fbound_propagator_soundness(const char* pred_name, F pred, bool test_completeness = true, bool disable_before_test = false) {
  std::string minval = "-20.0";
  std::string maxval = "20.0";
  std::vector<FItv> fitvs{create_float_interval<FItv>(minval, minval), create_float_interval<FItv>("-18.5", "18.5"), create_float_interval<FItv>(minval, "-3.333")
                  , create_float_interval<FItv>(minval, "-2.1234"), create_float_interval<FItv>(minval, "0.0"), create_float_interval<FItv>(minval, maxval)
                  , create_float_interval<FItv>("-2.6", "-1.1111"), create_float_interval<FItv>("-1.1111", "0.555")
                  , create_float_interval<FItv>("-2.2222", "2.2222"), create_float_interval<FItv>("-1.66", "-1.66"), create_float_interval<FItv>("0.0", "0.0")
                  , create_float_interval<FItv>("1.5", "1.5"), create_float_interval<FItv>("2.789", "2.789"), create_float_interval<FItv>("0.333", "1.111")
                  , create_float_interval<FItv>("1.456", "2.456"), create_float_interval<FItv>("0.5", maxval), create_float_interval<FItv>("2.5", maxval)
                  , create_float_interval<FItv>("3.333", maxval), create_float_interval<FItv>(maxval, maxval)
                  , create_float_interval<FItv>("1.5", "0.5"), create_float_interval<FItv>("2.1", "10.9999"), create_float_interval<FItv>("-25.123", "25.1234")
                  , create_float_interval<FItv>("-2.77777", "3.87665")};
   
  for(int i = 0; i < fitvs.size(); ++i) {
    for(int j = 0; j < fitvs.size(); ++j) {
      for(int k = 0; k < fitvs.size(); ++k) {
        FItv x = fitvs[i];
        FItv y = fitvs[j];
        FItv z = fitvs[k];

        std::string fzn = std::format("var {}..{}: x; var {}..{}: y; var {}..{}: z;\
          constraint {}(y, z, x);", x.lb().value(), x.ub().value(), y.lb().value(), y.ub().value(), z.lb().value(), z.ub().value(), pred_name);

        std::vector<double> xs, ys, zs;
        for(double a = x.lb().value(); a <= x.ub().value(); ++a) {
          for(double b = y.lb().value(); b <= y.ub().value(); ++b) {
            for(double c = z.lb().value(); c <= z.ub().value(); ++c) {
              if(pred(a, b, c)) {
                xs.push_back(a);
                ys.push_back(b);
                zs.push_back(c);
              }
            }
          }
        }
        FItv x2 = xs.empty() ? FItv::bot() : FItv(std::ranges::min(xs), std::ranges::max(xs));
        FItv y2 = ys.empty() ? FItv::bot() : FItv(std::ranges::min(ys), std::ranges::max(ys));
        FItv z2 = zs.empty() ? FItv::bot() : FItv(std::ranges::min(zs), std::ranges::max(zs));

        A a = create_and_interpret_and_tell<A, true, false>(fzn.data());
        fdeduce_and_test2(a, {x,y,z}, {x2,y2,z2}, test_completeness, disable_before_test);

        if(!a.is_bot()) {
          bool abstract_entailed = true;
          for(int i = 0; i < a.num_deductions(); ++i) {
            abstract_entailed &= a.ask(i, false);
          }
          bool is_bot = x2.is_bot() || y2.is_bot() || z2.is_bot();
          if(is_bot) {
            EXPECT_FALSE(abstract_entailed);
          }
          else {
            if(abstract_entailed) {
              for(double a = x2.lb().value(); a <= x2.ub().value(); ++a) {
                for(double b = y2.lb().value(); b <= y2.ub().value(); ++b) {
                  for(double c = z2.lb().value(); c <= z2.ub().value(); ++c) {
                    EXPECT_TRUE(pred(a, b, c)) << "The constraint is not entailed on (" << a << ", " << b << ", " << c << ") " << fzn;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <class A, class F>
void test_bound_propagator_completeness(const char* pred_name, F pred, bool with_fixpoint = false) {
  int minval = -5;
  int maxval = 5;
  for(int i = minval; i < maxval; ++i) {
    for(int j = minval; j < maxval; ++j) {
      for(int k = minval; k < maxval; ++k) {
        Itv x(i,i);
        Itv y(j,j);
        Itv z(k,k);

        std::string fzn = std::format("var {}..{}: x; var {}..{}: y; var {}..{}: z;\
          constraint {}(y, z, x);", x.lb().value(), x.ub().value(), y.lb().value(), y.ub().value(), z.lb().value(), z.ub().value(), pred_name);

        A a = create_and_interpret_and_tell<A, true>(fzn.data());
        if(with_fixpoint) {
          local::B has_changed = false;
          GaussSeidelIteration{}.fixpoint(
            a.num_deductions(),
            [&](size_t i) { return a.deduce(i); },
            has_changed);
        }
        bool abstract_entailed = true;
        for(int i = 0; i < a.num_deductions(); ++i) {
          abstract_entailed &= a.ask(i);
        }
        if(!pred(i,j,k)) {
          EXPECT_FALSE(abstract_entailed) << "The constraint is entailed on (" << i << ", " << j << ", " << k << ") but should not be." << fzn;
          deduce_and_test2(a, {x,y,z}, {Itv::bot(), Itv::bot(), Itv::bot()}, true, false);
          EXPECT_TRUE(a.is_bot());
        }
        else {
          EXPECT_TRUE(abstract_entailed) << "The constraint is not entailed on (" << i << ", " << j << ", " << k << ") but should be." << fzn;
          deduce_and_test2(a, {x,y,z}, {x,y,z}, true, false);
        }
      }
    }
  }
}

template <class A, class F>
void test_fbound_propagator_completeness(const char* pred_name, F pred, bool with_fixpoint = false) {
  double minval = -5.0;
  double maxval = 5.0;
  for(double i = minval; i < maxval; i+=1e-6) {
    for(double j = minval; j < maxval; j+=1e-6) {
      for(double k = minval; k < maxval; k+=1e-6) {
        FItv x = create_float_interval<FItv>(std::to_string(i), std::to_string(i));
        FItv y = create_float_interval<FItv>(std::to_string(j), std::to_string(j));
        FItv z = create_float_interval<FItv>(std::to_string(k), std::to_string(k));

        std::string fzn = std::format("var {}..{}: x; var {}..{}: y; var {}..{}: z;\
          constraint {}(y, z, x);", x.lb().value(), x.ub().value(), y.lb().value(), y.ub().value(), z.lb().value(), z.ub().value(), pred_name);

        A a = create_and_interpret_and_tell<A, true, false>(fzn.data());
        if(with_fixpoint) {
          local::B has_changed = false;
          GaussSeidelIteration{}.fixpoint(
            a.num_deductions(),
            [&](size_t i) { return a.deduce(i, false); },
            has_changed);
        }
        bool abstract_entailed = true;
        for(int i = 0; i < a.num_deductions(); ++i) {
          abstract_entailed &= a.ask(i, false);
        }
        if(!pred(i,j,k)) {
          EXPECT_FALSE(abstract_entailed) << "The constraint is entailed on (" << i << ", " << j << ", " << k << ") but should not be." << fzn;
          fdeduce_and_test2(a, {x,y,z}, {FItv::bot(), FItv::bot(), FItv::bot()}, true, false);
          EXPECT_TRUE(a.is_bot());
        }
        else {
          EXPECT_TRUE(abstract_entailed) << "The constraint is not entailed on (" << i << ", " << j << ", " << k << ") but should be." << fzn;
          fdeduce_and_test2(a, {x,y,z}, {x,y,z}, true, false);
        }
      }
    }
  }
}

template <class A, class F>
void test_bound_propagator_soundness_exhaustive(const char* pred_name, F pred, bool test_completeness = true) {
  int minval = -10;
  int maxval = 10;
  std::string fzn = std::format("var {}..{}: x; var {}..{}: y; var {}..{}: z; constraint {}(y, z, x);",
    minval, maxval, minval, maxval, minval, maxval, pred_name);
  A a = create_and_interpret_and_tell<A, true>(fzn.data());
  auto snap = a.snapshot();
  AVar xv(0,0);
  AVar yv(0,1);
  AVar zv(0,2);

  Itv x2, y2, z2;

  for(int xl = minval; xl <= maxval; ++xl) {
    for(int xu = xl; xu <= maxval; ++xu) {
      for(int yl = minval; yl <= maxval; ++yl) {
        for(int yu = yl; yu <= maxval; ++yu) {
          for(int zl = minval; zl <= maxval; ++zl) {
            for(int zu = zl; zu <= maxval; ++zu) {
              a.restore(snap);
              a.embed(xv, Itv(xl, xu));
              a.embed(yv, Itv(yl, yu));
              a.embed(zv, Itv(zl, zu));

              x2.meet_bot();
              y2.meet_bot();
              z2.meet_bot();
              for(int a = xl; a <= xu; ++a) {
                for(int b = yl; b <= yu; ++b) {
                  for(int c = zl; c <= zu; ++c) {
                    if(pred(a, b, c)) {
                      x2.join(Itv(a, a));
                      y2.join(Itv(b, b));
                      z2.join(Itv(c, c));
                    }
                  }
                }
              }
              // std::cout << a[0] << ", " << a[1] << ", " << a[2] << " --> "
              //           << x2 << ", " << y2 << ", " << z2 << std::endl;
              deduce_and_test2(a, {a[0],a[1],a[2]}, {x2,y2,z2}, test_completeness, false);

              if(!a.is_bot()) {
                bool abstract_entailed = true;
                for(int i = 0; i < a.num_deductions(); ++i) {
                  abstract_entailed &= a.ask(i);
                }
                bool is_bot = x2.is_bot() || y2.is_bot() || z2.is_bot();
                if(is_bot) {
                  EXPECT_FALSE(abstract_entailed);
                }
                else {
                  if(abstract_entailed) {
                    for(int a = x2.lb().value(); a <= x2.ub().value(); ++a) {
                      for(int b = y2.lb().value(); b <= y2.ub().value(); ++b) {
                        for(int c = z2.lb().value(); c <= z2.ub().value(); ++c) {
                          EXPECT_TRUE(pred(a, b, c)) << "The constraint is not entailed on (" << a << ", " << b << ", " << c << ") " << fzn;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
