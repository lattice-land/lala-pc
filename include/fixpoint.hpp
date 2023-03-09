// Copyright 2022 Pierre Talbot

#ifndef FIXPOINT_HPP
#define FIXPOINT_HPP

#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "memory.hpp"

namespace lala {

/** A simple form of sequential fixpoint computation based on Kleene fixpoint.
 * At each iteration, the refinement operations \f$ f_1, \ldots, f_n \f$ are simply composed by functional composition \f$ f = f_n \circ \ldots \circ f_1 \f$.
 * This strategy basically corresponds to the Gauss-Seidel iteration method. */
class GaussSeidelIteration {

public:
  CUDA void barrier() {}

  template <class A>
  CUDA void iterate(A& a, local::BInc& has_changed) {
    int n = a.num_refinements();
    for(int i = 0; !a.is_top() && i < n; ++i) {
      a.refine(i, has_changed);
    }
  }

  template <class A>
  CUDA void fixpoint(A& a, local::BInc& has_changed) {
    local::BInc changed(true);
    while(!(a.is_top()) && changed) {
      changed.dtell_bot();
      iterate(a, changed);
      has_changed.tell(changed);
    }
  }

  template <class A>
  CUDA local::BInc fixpoint(A& a) {
    local::BInc has_changed(false);
    fixpoint(a, has_changed);
    return has_changed;
  }
};

#ifdef __NVCC__

/** A simple form of fixpoint computation based on Kleene fixpoint.
 * At each iteration, the refinement operations \f$ f_1, \ldots, f_n \f$ are composed by parallel composition \f$ f = f_1 \| \ldots \| f_n \f$ meaning they are executed in parallel by different threads.
 * This is called an asynchronous iteration and it is due to (Cousot, Asynchronous iterative methods for solving a fixed point system of monotone equations in a complete lattice, 1977). */
template <class Allocator>
class AsynchronousIterationGPU {
public:
  using allocator_type = Allocator;
private:
  using atomic_binc = BInc<battery::AtomicMemoryBlock<allocator_type>>;
  battery::vector<atomic_binc, allocator_type> changed;
  battery::vector<atomic_binc, allocator_type> is_top;

  CUDA void assert_cuda_arch() {
    printf("AsynchronousIterationGPU must be used on the GPU device only.\n");
    assert(0);
  }

  CUDA void reset() {
    changed[0].tell_top();
    changed[1].dtell_bot();
    changed[2].dtell_bot();
    for(int i = 0; i < 3; ++i) {
      is_top[i].dtell_bot();
    }
  }

public:
  CUDA AsynchronousIterationGPU(const allocator_type& alloc = allocator_type()):
    changed(3, alloc), is_top(3, alloc)
  {}

  CUDA void barrier() {
  #ifndef __CUDA_ARCH__
      assert_cuda_arch();
  #else
    __syncthreads();
  #endif
  }

  template <class A, class M>
  CUDA void iterate(A& a, BInc<M>& has_changed) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
  #else
    int n = a.num_refinements();
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
      a.refine(t, has_changed);
    }
  #endif
  }

  template <class A, class M>
  CUDA void fixpoint(A& a, BInc<M>& has_changed) {
  #ifndef __CUDA_ARCH__
    assert_cuda_arch();
  #else
    reset();
    barrier();
    int i = 1;
    for(; !is_top[(i-1)%3] && changed[(i-1)%3]; ++i) {
      iterate(a, changed[i%3]);
      changed[(i+1)%3].dtell_bot(); // reinitialize changed for the next iteration.
      is_top[i%3].tell(a.is_top());
      barrier();
    }
    // It changes if we performed several iteration, or if the first iteration changed the abstract domain.
    has_changed.tell(changed[1]);
    has_changed.tell(changed[2]);
  #endif
  }
};

#endif

} // namespace lala

#endif
