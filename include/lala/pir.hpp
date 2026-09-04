// Copyright 2024 Pierre Talbot

#ifndef LALA_PC_PIR_HPP
#define LALA_PC_PIR_HPP

#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/root_ptr.hpp"
#include "battery/allocator.hpp"
#include "battery/algorithm.hpp"
#include "battery/bitset.hpp"

#include "lala/logic/logic.hpp"
#include "lala/logic/ternarize.hpp"
#include "lala/abstract_deps.hpp"
#include "lala/vstore.hpp"
#include "lala/zinterval.hpp"

namespace lala {

template <class A, class Alloc> class PIR;
namespace impl {
  template <class>
  struct is_pir_like {
    static constexpr bool value = false;
  };
  template<class A, class Alloc>
  struct is_pir_like<PIR<A, Alloc>> {
    static constexpr bool value = true;
  };
}

// This represents the constraints `X = Y [op] Z`.
struct bytecode_type {
  Sig op;
  AVar x;
  AVar y;
  AVar z;
  constexpr bytecode_type() = default;
  constexpr bytecode_type(const bytecode_type&) = default;
  CUDA INLINE const AVar& operator[](int i) const {
    return i == 0 ? x : (i == 1 ? y : z);
  }
};

/** PIR is an abstract transformer built on top of an abstract domain `A`.
    It is expected that `A` has a projection function `u = project(x)`.
    We also expect a `tell(x, u, has_changed)` function to join the abstract universe `u` in the domain of the variable `x`.
    An example of abstract domain satisfying these requirements is `VStore<Interval<ZInc>>`. */
template <class A, class Allocator = typename A::allocator_type>
class PIR {
public:
  using sub_type = A;
  using universe_type = typename A::universe_type;
  using local_universe_type = typename universe_type::basic_type;
  using allocator_type = Allocator;
  using sub_allocator_type = typename A::allocator_type;
  using this_type = PIR<sub_type, allocator_type>;

  template <class Alloc>
  struct snapshot_type
  {
    using sub_snap_type = A::template snapshot_type<Alloc>;
    int num_bytecodes;
    sub_snap_type sub_snap;

    CUDA snapshot_type(int num_bytecodes, sub_snap_type&& sub_snap)
      : num_bytecodes(num_bytecodes)
      , sub_snap(std::move(sub_snap))
    {}

    snapshot_type(const snapshot_type<Alloc>&) = default;
    snapshot_type(snapshot_type<Alloc>&&) = default;
    snapshot_type<Alloc>& operator=(snapshot_type<Alloc>&&) = default;
    snapshot_type<Alloc>& operator=(const snapshot_type<Alloc>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc& alloc = Alloc{})
      : num_bytecodes(other.num_bytecodes)
      , sub_snap(other.sub_snap, alloc)
    {}
  };

  using sub_ptr = abstract_ptr<sub_type>;

  constexpr static const char* name = "PIR";

  template <class A2, class Alloc2>
  friend class PIR;

  using bytecodes_type = battery::vector<bytecode_type, allocator_type>;
private:
  AType atype;
  sub_ptr sub;

  const local_universe_type ZERO;
  const local_universe_type ONE;

  static_assert(sizeof(int) == sizeof(AVar), "The size of AVar must be equal to the size of an int.");
  static_assert(sizeof(int) == sizeof(Sig), "The size of Sig must be equal to the size of an int.");

  using bytecodes_ptr = battery::root_ptr<battery::vector<bytecode_type, allocator_type>, allocator_type>;

  /** We represent the constraints X = Y [op] Z. */
  bytecodes_ptr bytecodes;

  /** When false, skip the operator-based sort in deduce(tell_type).
   *  Set to false when loading a pre-ordered TCN to preserve file order. */
  bool sort_bytecodes;

  using LB = typename local_universe_type::lb_type;
  using UB = typename local_universe_type::ub_type;

public:
  template <class Alloc, class SubType>
  struct interpreted_type {
    SubType sub_value;
    battery::vector<bytecode_type, Alloc> bytecodes;

    interpreted_type(interpreted_type&&) = default;
    interpreted_type& operator=(interpreted_type&&) = default;
    interpreted_type(const interpreted_type&) = default;

    CUDA interpreted_type(const SubType& sub_value, const Alloc& alloc = Alloc{})
      : sub_value(sub_value)
      , bytecodes(alloc)
    {}

    CUDA interpreted_type(const Alloc& alloc = Alloc{})
      : interpreted_type(SubType(alloc), alloc)
    {}

    template <class InterpretedType>
    CUDA interpreted_type(const InterpretedType& other, const Alloc& alloc = Alloc{})
      : sub_value(other.sub_value, alloc)
      , bytecodes(other.bytecodes, alloc)
    {}

    template <class Alloc2, class SubType2>
    friend struct interpreted_type;
  };

  template <class Alloc>
  using tell_type = interpreted_type<Alloc, typename sub_type::template tell_type<Alloc>>;

  template <class Alloc>
  using ask_type = interpreted_type<Alloc, typename sub_type::template ask_type<Alloc>>;

  CUDA PIR(AType atype, sub_ptr sub, const allocator_type& alloc = allocator_type{})
   : atype(atype), sub(std::move(sub))
   , ZERO(local_universe_type(0, 0))
   , ONE(local_universe_type(1, 1))
   , bytecodes(battery::allocate_root<bytecodes_type, allocator_type>(alloc, alloc))
   , sort_bytecodes(true)
  {}

  template <class PIR2>
  CUDA PIR(const PIR2& other, sub_ptr sub, const allocator_type& alloc = allocator_type{})
   : atype(atype), sub(sub)
   , ZERO(local_universe_type(0, 0))
   , ONE(local_universe_type(1, 1))
   , bytecodes(battery::allocate_root<bytecodes_type, allocator_type>(alloc, *(other.bytecodes), alloc))
   , sort_bytecodes(other.sort_bytecodes)
  {}

  CUDA PIR(PIR&& other)
    : atype(other.atype)
    , sub(std::move(other.sub))
    , ZERO(std::move(other.ZERO))
    , ONE(std::move(other.ONE))
    , bytecodes(std::move(other.bytecodes))
    , sort_bytecodes(other.sort_bytecodes)
  {}

private:
  // When activated (`deps.is_shared_copy()`), we avoid copying the propagators and share them with the ones of the root `other`.
  // This allows to save up memory and to avoid contention on L2 cache among blocks.
  template<class A2, class Alloc2, class... Allocators>
  CUDA static bytecodes_ptr init_bytecodes(const PIR<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps) {
    auto alloc = deps.template get_allocator<allocator_type>();
    if constexpr(std::is_same_v<allocator_type, Alloc2>) {
      if(deps.is_shared_copy()) {
        assert(static_cast<bool>(other.bytecodes));
        return other.bytecodes;
      }
    }
    bytecodes_ptr r = battery::allocate_root<bytecodes_type, allocator_type>(alloc, *(other.bytecodes), alloc);
    return std::move(r);
  }

public:
  template<class A2, class Alloc2, class... Allocators>
  CUDA PIR(const PIR<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , sub(deps.template clone<A>(other.sub))
   , ZERO(other.ZERO)
   , ONE(other.ONE)
   , bytecodes(init_bytecodes(other, deps))
   , sort_bytecodes(other.sort_bytecodes)
  {}

  /** Disable the operator-based sort applied after each tell.
   *  Call this before interpreting a pre-ordered TCN to preserve file order. */
  void disable_sort_bytecodes() {
    sort_bytecodes = false;
  }

  CUDA allocator_type get_allocator() const {
    return bytecodes.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  /** The underlying store of variables. */
  CUDA sub_ptr subdomain() const {
    return sub;
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return PIR{atype, battery::allocate_shared<sub_type>(sub_alloc, sub_type::bot(atype_sub, sub_alloc)), alloc};
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return PIR{atype, battery::allocate_shared<sub_type>(sub_alloc, sub_type::top(atype_sub, sub_alloc)), alloc};
  }

  template <class Env>
  CUDA static this_type bot(Env& env,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    AType atype_sub = env.extends_abstract_dom();
    AType atype = env.extends_abstract_dom();
    return bot(atype, atype_sub, alloc, sub_alloc);
  }

  template <class Env>
  CUDA static this_type top(Env& env,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    AType atype_sub = env.extends_abstract_dom();
    AType atype = env.extends_abstract_dom();
    return top(atype, atype_sub, alloc, sub_alloc);
  }

  /** Similar limitations than `PC::deduce`. */
  template <class Alloc2>
  CUDA local::B deduce(const tell_type<Alloc2>& t) {
    local::B has_changed = sub->deduce(t.sub_value);
    if(t.bytecodes.size() > 0) {
      bytecodes->reserve(bytecodes->size() + t.bytecodes.size());
      for(int i = 0; i < t.bytecodes.size(); ++i) {
        bytecodes->push_back(t.bytecodes[i]);
        if(t.bytecodes[i].op == EQ || t.bytecodes[i].op == LEQ) {
          sub->embed(t.bytecodes[i].x, local_universe_type(0,1));
        }
      }
    /** This is sorting the constraints `X = Y <op> Z` according to <OP>.
     * Note that battery::sorti is much slower than std::sort, therefore the #ifdef.
     * Skipped when sort_bytecodes is false (e.g. when loading a pre-ordered TCN). */
    if(sort_bytecodes) {
    #ifdef __CUDA_ARCH__
      battery::sorti(*bytecodes,
        [&](int i, int j) { return (*bytecodes)[i].op < (*bytecodes)[j].op; });
    #else
      std::stable_sort(bytecodes->data(), bytecodes->data() + bytecodes->size(),
        [](const bytecode_type& a, const bytecode_type& b) {
          // return a.op < b.op;
          return a.op == b.op ? (a.y.vid() == b.y.vid() ? (a.x.vid() == b.x.vid() ? a.z.vid() < b.z.vid() : a.x.vid() < b.x.vid()) : a.y.vid() < b.y.vid()) : a.op < b.op;
        });
    #endif
    }
      has_changed = true;
    }
    return has_changed;
  }

  CUDA bool embed(AVar x, const universe_type& dom) {
    return sub->embed(x, dom);
  }

  CUDA INLINE bytecode_type load_deduce(int i) const {
  #ifdef __CUDA_ARCH__
    // Vectorize load (int4).
    int4 b4 = reinterpret_cast<int4*>(bytecodes->data())[i];
    return *reinterpret_cast<bytecode_type*>(&b4);
  #else
    return (*bytecodes)[i];
  #endif
  }

  CUDA local::B ask(int i) const {
    return ask(load_deduce(i));
  }

  template <class Alloc2>
  CUDA local::B ask(const ask_type<Alloc2>& t) const {
    for(int i = 0; i < t.bytecodes.size(); ++i) {
      if(!ask(t.bytecodes[i])) {
        return false;
      }
    }
    return sub->ask(t.sub_value);
  }

  CUDA int num_deductions() const {
    return bytecodes->size();
  }

  CUDA local::B deduce(int i) {
    assert(i < num_deductions());
    return deduce(load_deduce(i));
  }

  using Itv = local_universe_type;

private:
  /** Deduce the constraint `x = y <op> z` by running lala-interval's propagator for `<op>`.
   * The propagators are bidirectional: they narrow all three intervals. */
  CUDA INLINE static void propagate(Sig op, Itv& r1, Itv& r2, Itv& r3) {
    switch(op) {
      case EQ:   tell::zreq(r1, r2, r3); break;
      case LEQ:  tell::zrleq(r1, r2, r3); break;
      case ADD:  tell::zadd(r1, r2, r3); break;
      case MUL:  tell::zmul(r1, r2, r3); break;
      case MIN:  tell::zmin(r1, r2, r3); break;
      case MAX:  tell::zmax(r1, r2, r3); break;
      case TDIV: tell::ztdiv(r1, r2, r3); break;
      case CDIV: tell::zcdiv(r1, r2, r3); break;
      case FDIV: tell::zfdiv(r1, r2, r3); break;
      case EDIV: tell::zediv(r1, r2, r3); break;
      default: assert(false);
    }
  }

  /** \return `true` when `x = y <op> z` is entailed by the current domains. */
  CUDA INLINE static bool entailed(Sig op, Itv& r1, Itv& r2, Itv& r3) {
    switch(op) {
      case EQ:   return ask::zreq(r1, r2, r3);
      case LEQ:  return ask::zrleq(r1, r2, r3);
      case ADD:  return ask::zadd(r1, r2, r3);
      case MUL:  return ask::zmul(r1, r2, r3);
      case MIN:  return ask::zmin(r1, r2, r3);
      case MAX:  return ask::zmax(r1, r2, r3);
      case TDIV: return ask::ztdiv(r1, r2, r3);
      case CDIV: return ask::zcdiv(r1, r2, r3);
      case FDIV: return ask::zfdiv(r1, r2, r3);
      case EDIV: return ask::zediv(r1, r2, r3);
      default: assert(false); return false;
    }
  }

  CUDA local::B ask(bytecode_type bytecode) const {
    Itv r1((*sub)[bytecode.x]);
    Itv r2((*sub)[bytecode.y]);
    Itv r3((*sub)[bytecode.z]);
    return entailed(bytecode.op, r1, r2, r3);
  }

public:
  CUDA local::B deduce(bytecode_type bytecode) {
    Itv r1((*sub)[bytecode.x]);
    Itv r2((*sub)[bytecode.y]);
    Itv r3((*sub)[bytecode.z]);
    propagate(bytecode.op, r1, r2, r3);
    local::B has_changed = sub->embed(bytecode.x, r1);
    has_changed |= sub->embed(bytecode.y, r2);
    has_changed |= sub->embed(bytecode.z, r3);
    return has_changed;
  }


  // Functions forwarded to the sub-domain `A`.

  /** `true` if the underlying abstract element is bot, `false` otherwise. */
  CUDA local::B is_bot() const {
    return sub->is_bot();
  }

  /** `true` if the underlying abstract element is top and there is no deduction function, `false` otherwise. */
  CUDA local::B is_top() const {
    return sub->is_top() && bytecodes->size() == 0;
  }

  CUDA auto /* universe_type or const universe_type& */ operator[](int x) const {
    return (*sub)[x];
  }

  CUDA auto /* universe_type or const universe_type& */ project(AVar x) const {
    return sub->project(x);
  }

  template <class Univ>
  CUDA void project(AVar x, Univ& u) const {
    sub->project(x, u);
  }

  CUDA int vars() const {
    return sub->vars();
  }

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    assert(static_cast<bool>(bytecodes));
    return snapshot_type<Alloc2>(bytecodes->size(), sub->snapshot(alloc));
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    int n = bytecodes->size();
    for(int i = snap.num_bytecodes; i < n; ++i) {
      bytecodes->pop_back();
    }
    sub->restore(snap.sub_snap);
  }

  /** An abstract element is extractable when it is not equal to bot, if all propagators are entailed and if the underlying abstract element is extractable. */
  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    if(is_bot()) {
      return false;
    }
    for(int i = 0; i < bytecodes->size(); ++i) {
      if(!ask(i)) {
        return false;
      }
    }
    return sub->is_extractable(strategy);
  }

  /** Extract the current element into `ua`.
   * \pre `is_extractable()` must be `true`.
   * For efficiency reason, if `B` is a propagator completion, the propagators are not copied in `ua`.
   *   (It is OK, since they are entailed, they don't bring information anymore.) */
  template <class B>
  CUDA void extract(B& ua) const {
    if constexpr(impl::is_pir_like<B>::value) {
      sub->extract(*ua.sub);
    }
    else {
      sub->extract(ua);
    }
  }
};

}

#endif
