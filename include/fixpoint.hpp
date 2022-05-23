// Copyright 2022 Pierre Talbot

#ifndef FIXPOINT_HPP
#define FIXPOINT_HPP

#include "z.hpp"

namespace lala {

template <class A>
void seq_fixpoint(A& a, BInc& has_changed) {
  BInc changed = BInc::top();
  while(neg(a.is_top()).guard() && changed.guard()) {
    changed.dtell(BInc::bot());
    if constexpr(std::is_same_v<void, decltype(std::declval<A>().refine(std::declval<BInc&>()))>)
    {
      a.refine(changed);
    }
    else {
      int n = a.num_refinements();
      for(int i = 0; neg(a.is_top()).guard() && i < n; ++i) {
        a.refine(i, changed);
      }
    }
    has_changed.tell(changed);
  }
}

template <class A>
BInc seq_fixpoint(A& a) {
  BInc has_changed = BInc::bot();
  seq_fixpoint(a, has_changed);
  return has_changed;
}

#ifdef __NVCC__

template <class A>
DEVICE void gpu_fixpoint(A& a, BInc& has_changed) {
  int tid = threadIdx.x;
  int stride = blockDim.x;
  __shared__ BInc changed[3] = {BInc::top(), BInc::bot(), BInc::bot()};
  int i;
  for(i = 1; neg(a.is_top()).guard() && changed[(i-1)%3].guard(); ++i) {
    for (int t = tid; t < a.num_refinements(); t += stride) {
      a.refine(t, changed[i%3]);
    }
    changed[(i+1)%3].dtell(BInc::bot());
    __syncthreads();
  }
  // If i == 1, we did not iterate at all, so has_changed remains unchanged.
  // If i == 2, we did only one iteration and if we did not reach top in `a`, then `a` did not change.
  // In all other cases, `a` was changed at least once.
  if((i == 2 && a.is_top().guard()) || i > 2) {
    has_changed.tell(BInc::top());
  }
}

template <class A>
DEVICE BInc gpu_fixpoint(A& a) {
  BInc has_changed = BInc::bot();
  gpu_fixpoint(a, has_changed);
  return has_changed;
}

#endif

} // namespace lala

#endif
