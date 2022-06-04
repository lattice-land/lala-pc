// Copyright 2022 Pierre Talbot

#ifndef FIXPOINT_HPP
#define FIXPOINT_HPP

#include "z.hpp"
#include "allocator.hpp"

namespace lala {

/** A simple form of sequential fixpoint computation based on Kleene fixpoint.
 * At each iteration, the refinement operations \f$ f_1, \ldots, f_n \f$ are simply composed by functional composition \f$ f = f_n \circ \ldots \circ f_1 \f$.
 * This strategy basically corresponds to the Gauss-Seidel iteration method. */
struct GaussSeidelIteration {
  CUDA static void barrier() {}

  template <class A>
  CUDA static void iterate(A& a, BInc& has_changed) {
    int n = a.num_refinements();
    for(int i = 0; !(a.is_top().guard()) && i < n; ++i) {
      a.refine(i, has_changed);
    }
  }

  template <class A>
  CUDA static void fixpoint(A& a, BInc& has_changed) {
    BInc changed = BInc::top();
    while(!(a.is_top().guard()) && changed.guard()) {
      changed.dtell(BInc::bot());
      iterate(a, changed);
      has_changed.tell(changed);
    }
  }

  template <class A>
  CUDA static BInc fixpoint(A& a) {
    BInc has_changed = BInc::bot();
    fixpoint(a, has_changed);
    return has_changed;
  }
};

/** A simple form of fixpoint computation based on Kleene fixpoint.
 * At each iteration, the refinement operations \f$ f_1, \ldots, f_n \f$ are composed by parallel composition \f$ f = f_1 \| \ldots \| f_n \f$ meaning they are executed in parallel by different threads.
 * This is called an asynchronous iteration and it is due to (Cousot, Asynchronous iterative methods for solving a fixed point system of monotone equations in a complete lattice, 1977). */
struct AsynchronousIterationGPU {
  CUDA static void barrier() {
    #ifndef __CUDA_ARCH__
      assert(0);
    #else
      __syncthreads();
    #endif
  }

  template <class A>
  CUDA static void iterate(A& a, BInc& has_changed) {
    #ifndef __CUDA_ARCH__
      assert(0);
    #else
      int n = a.num_refinements();
      for (int t = threadIdx.x; t < n; t += blockDim.x) {
        a.refine(t, has_changed);
      }
    #endif
  }

  template <class A>
  CUDA static void fixpoint(A& a, BInc& has_changed) {
    #ifndef __CUDA_ARCH__
      assert(0);
    #else
      __shared__ int shared_mem[sizeof(BInc) * 6];
      BInc* changed[3];
      BInc* is_top[3];
      if(threadIdx.x == 0) {
        PoolAllocator alloc(shared_mem, sizeof(shared_mem));
        changed[0] = new(alloc) BInc(BInc::top());
        for(int i = 1; i < 3; ++i) {
          changed[i] = new(alloc) BInc(BInc::bot());
        }
        for(int i = 0; i < 3; ++i) {
          is_top[i] = new(alloc) BInc(BInc::bot());
        }
      }
      barrier();
      int n = a.num_refinements();
      for(i = 1; !(is_top[(i-1)%3]->guard()) && changed[(i-1)%3]->guard(); ++i) {
        for (int t = threadIdx.x; t < n; t += blockDim.x) {
          a.refine(t, changed[i%3]);
        }
        changed[(i+1)%3]->dtell(BInc::bot());
        is_top[i%3]->tell(a.is_top());
        barrier();
      }
      // If i == 1, we did not iterate at all, so has_changed remains unchanged.
      // If i == 2, we did only one iteration and if we did not reach top in `a`, then `a` did not change.
      // In all other cases, `a` was changed at least once.
      if((i == 2 && a.is_top().guard()) || i > 2) {
        has_changed.tell(BInc::top());
      }
    #endif
  }
};

} // namespace lala

#endif
