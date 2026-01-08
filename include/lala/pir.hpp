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
#include "battery/utility.hpp"

#include "lala/logic/logic.hpp"
#include "lala/logic/ternarize.hpp"
#include "lala/universes/arith_bound.hpp"
#include "lala/abstract_deps.hpp"
#include "lala/vstore.hpp"

#include "terms.hpp"

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
  using local_universe_type = typename universe_type::local_type;
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

  constexpr static const bool is_abstract_universe = false;
  constexpr static const bool sequential = sub_type::sequential;
  constexpr static const bool is_totally_ordered = false;
  constexpr static const bool preserve_bot = true;
  // The next properties should be checked more seriously, relying on the sub-domain might be uneccessarily restrictive.
  constexpr static const bool preserve_top = sub_type::preserve_top;
  constexpr static const bool preserve_join = sub_type::preserve_join;
  constexpr static const bool preserve_meet = sub_type::preserve_meet;
  constexpr static const bool injective_concretization = sub_type::injective_concretization;
  constexpr static const bool preserve_concrete_covers = sub_type::preserve_concrete_covers;
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

  using LB = typename local_universe_type::LB;
  using UB = typename local_universe_type::UB;

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
   , ZERO(local_universe_type::eq_zero())
   , ONE(local_universe_type::eq_one())
   , bytecodes(battery::allocate_root<bytecodes_type, allocator_type>(alloc, alloc))
  {}

  template <class PIR2>
  CUDA PIR(const PIR2& other, sub_ptr sub, const allocator_type& alloc = allocator_type{})
   : atype(atype), sub(sub)
   , ZERO(local_universe_type::eq_zero())
   , ONE(local_universe_type::eq_one())
   , bytecodes(battery::allocate_root<bytecodes_type, allocator_type>(alloc, *(other.bytecodes), alloc))
  {}

  CUDA PIR(PIR&& other)
    : atype(other.atype)
    , sub(std::move(other.sub))
    , ZERO(std::move(other.ZERO))
    , ONE(std::move(other.ONE))
    , bytecodes(std::move(other.bytecodes))
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
  {}

  CUDA allocator_type get_allocator() const {
    return bytecodes.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
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

private:
  /** We interpret the formula `f` in the value `intermediate`, note that we only add one constraint to `intermediate` if the interpretation succeeds. */
  template <IKind kind, bool diagnose, class F, class Env, class Intermediate>
  CUDA bool interpret_formula(const F& f, Env& env, Intermediate& intermediate, IDiagnostics& diagnostics) const {
    if(f.type() != aty() && !f.is_untyped()) {
      RETURN_INTERPRETATION_ERROR("The type of the formula does not match the type of this abstract domain.");
    }
    if(f.is_binary()) {
      Sig sig = f.sig();
      // Expect constraint of the form X = Y <OP> Z, or Y <OP> Z = X.
      int left = f.seq(0).is_binary() ? 1 : 0;
      int right = f.seq(1).is_binary() ? 1 : 0;
      if((sig == EQ || sig == EQUIV)  && (left + right == 1)) {
        auto& X = f.seq(left);
        auto& Y = f.seq(right).seq(0);
        auto& Z = f.seq(right).seq(1);
        bytecode_type bytecode;
        bytecode.op = f.seq(right).sig();
        if(X.is_variable() && Y.is_variable() && Z.is_variable() &&
          (bytecode.op == ADD || bytecode.op == MUL || ::lala::is_z_division(bytecode.op) || bytecode.op == EMOD
          || bytecode.op == MIN || bytecode.op == MAX
          || bytecode.op == EQ || bytecode.op == LEQ))
        {
          if( env.template interpret<diagnose>(X, bytecode.x, diagnostics)
           && env.template interpret<diagnose>(Y, bytecode.y, diagnostics)
           && env.template interpret<diagnose>(Z, bytecode.z, diagnostics))
          {
            intermediate.bytecodes.push_back(bytecode);
            return true;
          }
          RETURN_INTERPRETATION_ERROR("Could not interpret the variables in the environment.");
        }
      }
    }
    RETURN_INTERPRETATION_ERROR("The shape of this formula is not supported.");
  }

public:
  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    size_t error_context = 0;
    if constexpr(diagnose) {
      diagnostics.add_suberror(IDiagnostics(false, name, "Uninterpretable formula in both PIR and its sub-domain.", f));
      error_context = diagnostics.num_suberrors();
    }
    bool res = false;
    AType current = f.type();
    const_cast<F&>(f).type_as(sub->aty()); // We will restore the type after the call to sub->interpret.
    if(sub->template interpret<kind, diagnose>(f, env, intermediate.sub_value, diagnostics)) {
      res = true;
    }
    const_cast<F&>(f).type_as(current);
    if(!res) {
      res = interpret_formula<kind, diagnose>(f, env, intermediate, diagnostics);
    }
    if constexpr(diagnose) {
      diagnostics.merge(res, error_context);
    }
    return res;
  }

  /** PIR expects a non-conjunctive formula \f$ c \f$ which can either be interpreted in the sub-domain `A` or in the current domain.
  */
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    return interpret<IKind::TELL, diagnose>(f, env, tell, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc2>& ask, IDiagnostics& diagnostics) const {
    return interpret<IKind::ASK, diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
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
     * Note that battery::sorti is much slower than std::sort, therefore the #ifdef. */
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
      has_changed = true;
    }
    return has_changed;
  }

  template <class Alloc2>
  CUDA local::B fdeduce(const tell_type<Alloc2>& t) {
    local::B has_changed = sub->fdeduce(t.sub_value);
    if(t.bytecodes.size() > 0) {
      bytecodes->reserve(bytecodes.size() + t.bytecodes.size());
      for(int i = 0; i < t.bytecodes.size(); ++i) {
        bytecodes->push_back(t.bytecodes[i]);
        if(t.bytecodes[i].op == EQ || t.bytecodes.op == LEQ) {
          sub->embed(t.bytecodes[i].x, local_universe_type(0,1));
        }
      }
    #ifdef __CUDA_ARCH__
      battery::sorti(*bytecodes,
        [&](int i, int j) { return (*bytecodes)[i].op < (*bytecodes)[j].op; });
    #else 
      std::stable_sort(bytecodes->data(), bytecodes->data() + bytecodes->size(),
        [](const bytecode_type& a, const bytecode_type&b ) {
          return a.op == b.op ? (a.y.vid() == b.y.vid() ? (a.x.vid() == b.x.vid() ? a.z.vid() < b.z.vid() : a.x.vid() < b.x.vid()) : a.y.vid() < b.y.vid()) : a.op < b.op;
        });
    #endif 
      has_changed = true;
    }
    return has_changed;
  }

  CUDA bool embed(AVar x, const universe_type& dom) {
    return sub->embed(x, dom);
  }

public:
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

public:
  CUDA local::B deduce(int i) {
    assert(i < num_deductions());
    return deduce(load_deduce(i));
  }

  CUDA local::B fdeduce(int i) {
    assert(i < num_deductions());
    return fdeduce(load_deduce(i));
  }

  using Itv = local_universe_type;
  using value_t = typename Itv::LB::value_type;

// Some defines to make the code more readable, and closer from the paper, without introducing local variables.
#define xl r1.lb().value()
#define xu r1.ub().value()
#define yl r2.lb().value()
#define yu r2.ub().value()
#define zl r3.lb().value()
#define zu r3.ub().value()

#define INF std::numeric_limits<value_t>::max()
#define MINF std::numeric_limits<value_t>::min()

private:
  CUDA INLINE value_t div(value_t a, Sig op, value_t b) const {
    switch(op) {
      case TDIV: return battery::tdiv(a, b);
      case CDIV: return battery::cdiv(a, b);
      case FDIV: return battery::fdiv(a, b);
      case EDIV: return battery::ediv(a, b);
      default: assert(false); return a;
    }
  }

  CUDA local::B ask(bytecode_type bytecode) const {
    // We load the variables.
    local_universe_type r1((*sub)[bytecode.x]);
    local_universe_type r2((*sub)[bytecode.y]);
    local_universe_type r3((*sub)[bytecode.z]);
    switch(bytecode.op) {
      case EQ: return (xl == 1 && yu == zl && yl == zu) || (xu == 0 && (yu < zl || yl > zu));
      case LEQ: return (xl == 1 && yu <= zl) || (xu == 0 && yl > zu);
      case ADD: return (xl == xu && yl == yu && zl == zu && xl == yl + zl);
      case MUL: return xl == xu &&
                        ((yl == yu && zl == zu && xl == yl * zl)
                      || (xl == 0 && (r2 == 0 || r3 == 0)));
      case TDIV:
      case CDIV:
      case FDIV:
      case EDIV: return (xl == xu && yl == yu && zl == zu && zl != 0 && xl == div(yl, bytecode.op, zl))
                     || (xl == yu && xu == yl && xl == 0 && (zl > 0 || zu < 0)); // 0 = 0 / z (z != 0).
      case EMOD: return (xl == xu && yl == yu && zl == zu && zl != 0 && xl == battery::emod(yl, zl));
      case MIN: return (xl == yu && xu == yl && yu <= zl) || (xl == zu && xu == zl && zu <= yl);
      case MAX: return (xl == yu && xu == yl && yl >= zu) || (xl == zu && xu == zl && zl >= yu);
      default: assert(false); return false;
    }
  }

  CUDA INLINE value_t min(value_t a, value_t b) const {
    return battery::min(a, b);
  }

  CUDA INLINE value_t max(value_t a, value_t b) const {
    return battery::max(a, b);
  }

  // r1 = r2 / r3
  CUDA INLINE void itv_div(Sig op, Itv& r1, Itv& r2, Itv& r3) const {
    if(zl < 0 && zu > 0) {
      r1.lb() = max(xl, min(yl, yu == MINF ? INF : -yu));
      r1.ub() = min(xu, max(yl == INF ? MINF : -yl, yu));
    }
    else {
      if(zl == 0) { r3.lb() = 1; }
      if(zu == 0) { r3.ub() = -1; }
      if(yl == MINF || yu == INF || zl == MINF || zu == INF) { return; }
      // Although it usually does not hurt to compute with bottom values, in this case, we want to prevent it from being equal to 0 (the previous conditions suppose r3 != bot).
      if(r3.is_bot()) { return; }
      auto t1 = div(yl, op, zl);
      auto t2 = div(yl, op, zu);
      auto t3 = div(yu, op, zl);
      auto t4 = div(yu, op, zu);
      r1.lb() = max(xl, min(min(t1, t2), min(t3, t4)));
      r1.ub() = min(xu, max(max(t1, t2), max(t3, t4)));
    }
  }

  CUDA INLINE Itv num_fdiv(const Itv& r1, const Itv& r3) const {
    if(zl < 0 && zu > 0) {
      return Itv(min(min(xl, -xu), min(xl * zu, (xu + 1) * zl + 1)),
                 max(max(-xl, xu), max(xl * zl, (xu + 1) * zu - 1)));
    }
    else if(zl > 0 || zu < 0) {
      return Itv(min(min(xl * zl, xl * zu), min((xu + 1) * zl + 1, (xu + 1) * zu + 1)),
                 max(max(xl * zl, xl * zu), max((xu + 1) * zl - 1, (xu + 1) * zu - 1)));
    }
    return Itv::top();
  }

  CUDA INLINE Itv num_cdiv(const Itv& r1, const Itv& r3) const {
    if(zl < 0 && zu > 0) {
      return Itv(min(min(xl, -xu), min(xu * zl, (xl - 1) * zu + 1)),
                 max(max(-xl, xu), max(xu * zu, (xl - 1) * zl - 1)));
    }
    else if(zl > 0 || zu < 0) {
      return Itv(min(min(xu * zl, xu * zu), min((xl - 1) * zl + 1, (xl - 1) * zu + 1)),
                 max(max(xu * zl, xu * zu), max((xl - 1) * zl - 1, (xl - 1) * zu - 1)));
    }
    return Itv::top();
  }

  CUDA INLINE Itv num_tdiv(const Itv& r1, const Itv& r3) const {
    if(xl > 0) {
      return num_fdiv(r1, r3);
    }
    else if(xu < 0) {
      return num_cdiv(r1, r3);
    }
    else if(xl <= 0 && 0 <= xu) {
      Itv r(min(zl, -zu) + 1, max(-zl, zu) - 1);
      if(xl != 0) { r.join(num_cdiv(Itv(xl,-1), r3)); }
      if(xu != 0) { r.join(num_fdiv(Itv(1, xu), r3)); }
      return r;
    }
    return Itv::top();
  }

  // Lemma C.10
  CUDA INLINE Itv num_ediv(const Itv& r1, const Itv& r3) const {
    if(zl > 0) { return num_fdiv(r1, r3); }
    else if(zu < 0) { return num_cdiv(r1, r3); }
    else if(zl < 0 && zu > 0) {
      return fjoin(num_cdiv(r1, Itv(zl, -1)), num_fdiv(r1, Itv(1, zu)));
    }
    return Itv::top();
  }

  // Lemma A.11
  CUDA Itv den_fdiv(const Itv& r1, const Itv& r2) const {
    using namespace battery;
    if(xl > 0 || xu + 1 < 0) {
      if(yl > 0) {
        return Itv(
          min(fdiv(yl, xu + 1), fdiv(yu, xu + 1)) + 1,
          max(fdiv(yl, xl), fdiv(yu, xl))
        );
      }
      else if(yu < 0) {
        return Itv(
          min(cdiv(yl, xl), cdiv(yu, xl)),
          max(cdiv(yl, xu + 1), cdiv(yu, xu + 1)) - 1
        );
      }
      else if(0 == yl && yl < yu) {
        return den_fdiv(r1, Itv(1, yu));
      }
      else if(yl < yu && yu == 0) {
        return den_fdiv(r1, Itv(yl, -1));
      }
      else if(yl < 0 && 0 < yu) {
        return fjoin(den_fdiv(r1, Itv(yl, -1)), den_fdiv(r1, Itv(1, yu)));
      }
      else if(yl == 0 && yu == 0) {
        return Itv::bot();
      }
    }
    else if(xl == 0 && xu == 0) {
      if(yl > 0) { return Itv(yl + 1, INF); }
      else if(yu < 0) { return Itv(MINF, yu - 1); }
      // else if(yl <= 0 && 0 <= yu) { return Itv::top(); }
    }
    else if(xl == -1 && xu == -1) {
      if(yl > 0) { return Itv(MINF, -yl); }
      else if(yu < 0) { return Itv(-yu, INF); }
      else if(0 == yl && yl < yu) { return Itv(MINF, -1); }
      else if(yl < yu && yu == 0) { return Itv(1, INF); }
      else if(yl == 0 && yu == 0) { return Itv::bot(); }
    }
    else if(xl == 0 && 0 < xu) {
      return fjoin(den_fdiv(Itv(0,0), r2), den_fdiv(Itv(1, xu), r2));
    }
    else if(xl < -1 && xu == -1) {
      return fjoin(den_fdiv(Itv(xl, -2), r2), den_fdiv(Itv(-1, -1), r2));
    }
    else if(xl <= -1 && xu >= 0) {
      Itv r(den_fdiv(Itv(-1, -1), r2));
      r.join(den_fdiv(Itv(0, 0), r2));
      if(xl != -1) { r.join(den_fdiv(Itv(xl, -2), r2)); }
      if(xu != 0) { r.join(den_fdiv(Itv(1, xu), r2)); }
      return r;
    }
    return Itv::top();
  }

  // Lemma C.12
  CUDA Itv den_cdiv(const Itv& r1, const Itv& r2) const {
    using namespace battery;
    if(xl - 1 > 0 || xu < 0) {
      if(yl > 0) {
        return Itv(
          min(cdiv(yl, xu), cdiv(yu, xu)),
          max(cdiv(yl, xl - 1), cdiv(yu, xl - 1)) - 1
        );
      }
      else if(yu < 0) {
        return Itv(
          min(fdiv(yl, xl - 1), fdiv(yu, xl - 1)) + 1,
          max(fdiv(yl, xu), fdiv(yu, xu))
        );
      }
      else if(0 == yl && yl < yu) {
        return den_cdiv(r1, Itv(1, yu));
      }
      else if(yl < yu && yu == 0) {
        return den_cdiv(r1, Itv(yl, -1));
      }
      else if(yl < 0 && 0 < yu) {
        return fjoin(den_cdiv(r1, Itv(yl, -1)), den_cdiv(r1, Itv(1, yu)));
      }
    }
    else if(xl == 0 && xu == 0) {
      if(yl > 0) { return Itv(MINF, -yl + 1); }
      else if(yu < 0) { return Itv(-yu + 1, INF); }
    }
    else if(xl == -1 && xu == -1) {
      if(yl > 0) { return Itv(yl, INF); }
      else if(yu < 0) { return Itv(MINF, yu); }
      else if(0 == yl && yl < yu) { return Itv(1, INF); }
      else if(yl < yu && yu == 0) { return Itv(MINF, -1); }
    }
    else if(xl < 0 && xl == xu) {
      return fjoin(den_cdiv(Itv(xl, -1), r2), den_cdiv(Itv(0, 0), r2));
    }
    else if(xl == 1 && xl < xu) {
      return fjoin(den_cdiv(Itv(1, 1), r2), den_cdiv(Itv(2, xu), r2));
    }
    else if(xl <= 0 && xu >= 1) {
      Itv r(den_cdiv(Itv(1, 1), r2));
      r.join(den_cdiv(Itv(0, 0), r2));
      if(xl != 0) { r.join(den_cdiv(Itv(xl, -1), r2)); }
      if(xu != 1) { r.join(den_cdiv(Itv(2, xu), r2)); }
      return r;
    }
    return Itv::top();
  }

  // Lemma C.13
  CUDA Itv den_tdiv(const Itv& r1, const Itv& r2, const Itv& r3) const {
    if(xl > 0) { return den_fdiv(r1, r2); }
    else if(xu < 0) { return den_cdiv(r1, r2); }
    else if(xl == 0 && xu == 0) {
      if(yl > 0 && zl > 0) { return Itv(yl + 1, INF); }
      if(yl > 0 && zu < 0) { return Itv(MINF, -yl - 1); }
      if(yu < 0 && zl > 0) { return Itv(-yu + 1, INF); }
      if(yu < 0 && zu < 0) { return Itv(MINF, yu - 1); }
    }
    else if(xl <= 0 && 0 <= xu) {
      Itv r(den_tdiv(Itv(0, 0), r2, r3));
      if(xl != 0) { r.join(den_cdiv(Itv(xl, -1), r2)); }
      if(xu != 0) { r.join(den_fdiv(Itv(1, xu), r2)); }
      return r;
    }
    return Itv::top();
  }

  CUDA INLINE Itv den_ediv(const Itv& r1, const Itv& r2, const Itv& r3) const {
    if(zl > 0) { return den_fdiv(r1, r2); }
    else if(zu < 0) { return den_cdiv(r1, r2); }
    else if(zl < 0 && 0 < zu) {
      return fjoin(den_fdiv(r1, r2), den_cdiv(r1, r2));
    }
    return Itv::top();
  }

  CUDA INLINE void itv_div_num(Sig op, Itv& r1, Itv& r2, Itv& r3) const {
    switch(op) {
      case FDIV: {
        r2.meet(num_fdiv(r1, r3));
        break;
      }
      case CDIV: {
        r2.meet(num_cdiv(r1, r3));
        break;
      }
      case TDIV: {
        r2.meet(num_tdiv(r1, r3));
        break;
      }
      case EDIV: {
        r2.meet(num_ediv(r1, r3));
        break;
      }
    }
  }

  CUDA INLINE void itv_div_den(Sig op, Itv& r1, Itv& r2, Itv& r3) const {
    switch(op) {
      case FDIV: {
        r3.meet(den_fdiv(r1, r2));
        break;
      }
      case CDIV: {
        r3.meet(den_cdiv(r1, r2));
        break;
      }
      case TDIV: {
        r3.meet(den_tdiv(r1, r2, r3));
        break;
      }
      case EDIV: {
        r3.meet(den_ediv(r1, r2, r3));
        break;
      }
    }
  }

  CUDA INLINE void mul_inv(const Itv& r1, Itv& r2, Itv& r3) {
    if(xl > 0 || xu < 0) {
      printf("we're in the first mul_inv case.\n");
      if(zl == 0) { r3.lb() = 1; }
      if(zu == 0) { r3.ub() = -1; }
    }
    if((xl > 0 || xu < 0) && zl < 0 && zu > 0) {
      printf("we're in the second mul_inv case.\n");
      r2.lb() = max(yl, min(xl, xu == MINF ? INF : -xu));
      r2.ub() = min(yu, max(xl == INF ? MINF : -xl, xu));
    }
    else if(xl > 0 || xu < 0 || zl > 0 || zu < 0) {
      printf("we're in the third mul_inv case1.\n");
      if(xl == MINF || xu == INF || zl == MINF || zu == INF) { return; }
      // Although it usually does not hurt to compute with bottom values, in this case, we want to prevent it from being equal to 0 (the previous conditions suppose r3 != bot).
      printf("we're in the third mul_inv case2.\n");
      if(r3.is_bot()) { return; }
      printf("we're in mul_inv to update r2 value.\n");
      r2.lb() = max(yl, min(min(battery::cdiv(xl, zl), battery::cdiv(xl, zu)), min(battery::cdiv(xu, zl), battery::cdiv(xu, zu))));
      r2.ub() = min(yu, max(max(battery::fdiv(xl, zl), battery::fdiv(xl, zu)), max(battery::fdiv(xu, zl), battery::fdiv(xu, zu))));
    }
  }

public:
  CUDA local::B deduce(bytecode_type bytecode) {
    local::B has_changed = false;
    // We load the variables.
    Itv r1((*sub)[bytecode.x]);
    Itv r2((*sub)[bytecode.y]);
    Itv r3((*sub)[bytecode.z]);
    value_t t1, t2, t3, t4; // Temporary variables for multiplication.

    switch(bytecode.op) {
      case EQ: {
        if(r1 == ONE) {
          has_changed |= sub->embed(bytecode.y, r3);
          has_changed |= sub->embed(bytecode.z, r2);
        }
        else if(r1 == ZERO && (yl == yu || zl == zu)) {
          has_changed |= sub->embed(zl == zu ? bytecode.y : bytecode.z, // If z is a singleton, we update y, and vice-versa.
            Itv(
              yl == zl ? yl + 1 : LB::top().value(),
              yu == zu ? yu - 1 : UB::top().value()));
        }
        else if(yu == zl && yl == zu) { has_changed |= sub->embed(bytecode.x, ONE); }
        else if(yl > zu || yu < zl) { has_changed |= sub->embed(bytecode.x, ZERO); }
        return has_changed;
      }
      case LEQ: {
        if(r1 == ONE) {
          has_changed |= sub->embed(bytecode.y, Itv(yl, zu));
          has_changed |= sub->embed(bytecode.z, Itv(yl, zu));
        }
        else if(r1 == ZERO) {
          has_changed |= sub->embed(bytecode.y, Itv(zl + 1, yu));
          has_changed |= sub->embed(bytecode.z, Itv(zl, yu - 1));
        }
        else if(yu <= zl) { has_changed |= sub->embed(bytecode.x, ONE); }
        else if(yl > zu) { has_changed |= sub->embed(bytecode.x, ZERO); }
        return has_changed;
      }
      case ADD: {
        r1.lb() = (yl == MINF || zl == MINF) ? xl : max(xl, yl + zl);
        r1.ub() = (yu == INF || zu == INF) ? xu : min(xu, yu + zu);
        r2.lb() = (xl == MINF || zu == INF) ? yl : max(yl, xl - zu);
        r2.ub() = (xu == INF || zl == MINF) ? yu : min(yu, xu - zl);
        r3.lb() = (xl == MINF || yu == INF) ? zl : max(zl, xl - yu);
        r3.ub() = (xu == INF || yl == MINF) ? zu : min(zu, xu - yl);
        break;
      }
      case MUL: {
        if(yl != MINF && yu != INF && zl != MINF && zu != INF) {
          t1 = yl * zl;
          t2 = yl * zu;
          t3 = yu * zl;
          t4 = yu * zu;
          r1.lb() = max(xl, min(min(t1, t2), min(t3, t4)));
          r1.ub() = min(xu, max(max(t1, t2), max(t3, t4)));
        }
        mul_inv(r1, r2, r3);
        mul_inv(r1, r3, r2);
        break;
      }
      case TDIV:
      case CDIV:
      case FDIV:
      case EDIV: {
        itv_div(bytecode.op, r1, r2, r3);
        if(!r1.is_bot() && !r3.is_bot()) {
          itv_div_num(bytecode.op, r1, r2, r3);
          if(!r2.is_bot()) {
            itv_div_den(bytecode.op, r1, r2, r3);
          }
        }
        break;
      }
      case EMOD: {
        if(zl == 0) { r3.lb() = 1; }
        if(zu == 0) { r3.ub() = -1; }
        if(yl == yu && zl == zu) {
          r1.lb() = battery::emod(yl, zl);
          r1.ub() = xl;
        }
        break;
      }
      case MIN: {
        r1.lb() = max(xl, min(yl, zl));
        r1.ub() = min(xu, min(yu, zu));
        r2.lb() = max(yl, xl);
        if(xu < zl) { r2.ub() = min(yu, xu); }
        r3.lb() = max(zl, xl);
        if(xu < yl) { r3.ub() = min(zu, xu); }
        break;
      }
      case MAX: {
        r1.lb() = max(xl, max(yl, zl));
        r1.ub() = min(xu, max(yu, zu));
        r2.ub() = min(yu, xu);
        if(xl > zu) { r2.lb() = max(yl, xl); }
        r3.ub() = min(zu, xu);
        if(xl > yu) { r3.lb() = max(zl, xl); }
        break;
      }
      default: assert(false);
    }
    has_changed |= sub->embed(bytecode.x, r1);
    has_changed |= sub->embed(bytecode.y, r2);
    has_changed |= sub->embed(bytecode.z, r3);
    return has_changed;
  }

  CUDA local::B fdeduce(bytecode_type bytecode) {
    local::B has_changed = false;
    // We load the variables. 
    Itv r1((*sub)[bytecode.x]);
    Itv r2((*sub)[bytecode.y]);
    Itv r3((*sub)[bytecode.z]);
    value_t t1, t2, t3, t4, t5, t6, t7, t8; // Temporary variables for multipilication. 
    switch(bytecode.op) {
      case EQ: {
        if(r1 == ONE) {
          has_changed |= sub->embed(bytecode.y, r3);
          has_changed |= sub->embed(bytecode.z, r2);
        }
        else if(r1 == ZERO && (yl == yu || zl == zu)) { // If z is a singleton, we update y, and vice-versa.
          // has_changed |= sub->embed(zl == zu ? bytecode.y : bytecode.z,
          // Itv(
          //   yl == zl ? yl + MINF : LB::top().value(),
          //   yu == zu ? yu - MINF : UB::top().value()));
        }
        else if (yu == zl && yl == zu) { has_changed |= sub->embed(bytecode.x, ONE); } // ? not sure why yet.
        else if (yl > zu || yu < zl) { has_changed |= sub->embed(bytecode.x, ZERO); }
        return has_changed;
      }
      case LEQ: {
        if(r1 == ONE) {
          has_changed |= sub->embed(bytecode.y, Itv(yl, zu));
          has_changed |= sub->embed(bytecode.z, Itv(yl, zu));
        }
        else if(r1 == ZERO) { 
          // has_changed |= sub->embed(bytecode.y, Itv(zl + MINF, yu));
          // has_changed |= sub->embed(bytecode.z, Itv(zl, yu - MINF));
        }
        else if(yu <= zl) { has_changed |= sub->embed(bytecode.x, ONE); }
        else if(yl > zu) { has_changed |= sub->embed(bytecode.x, ZERO); }
        return has_changed;
      }
      case ADD: {
        r1.lb() = (yl == MINF || zl == MINF) ? xl : max(xl, battery::add_down(yl, zl));
        r1.ub() = (yu == INF || zu == INF) ? xu : min(xu, battery::add_up(yu, zu));
        r2.lb() = (xl == MINF || zu == INF) ? yl : max(yl, battery::sub_down(xl, zu));
        r2.ub() = (xu == INF || zl == MINF) ? yu : min(yu, battery::sub_up(xu, zl));
        r3.lb() = (xl == MINF || yu == INF) ? zl : max(zl, battery::sub_down(xl, yu));
        r3.ub() = (xu == INF || yl == MINF) ? zu : min(zu, battery::sub_up(xu, yl));
        break;
      }
      case MUL: {
        printf("before mul: xl = %f, xu = %f, yl = %f, yu = %f, zl = %f, zu = %f\n", xl, xu , yl, yu, zl, zu);
        if(yl != MINF && yu != INF && zl != MINF && zu != INF) {
          t1 = battery::mul_down(yl, zl);
          t2 = battery::mul_down(yl, zu);
          t3 = battery::mul_down(yu, zl);
          t4 = battery::mul_down(yu, zu);
          t5 = battery::mul_up(yl, zl);
          t6 = battery::mul_up(yl, zu);
          t7 = battery::mul_up(yu, zl);
          t8 = battery::mul_up(yu, zu);
          r1.lb() = max(xl, min(min(t1, t2), min(t3, t4)));
          r1.ub() = min(xu, max(max(t5, t6), max(t7, t8)));
        }
        printf("after updating r1: xl = %f, xu = %f, yl = %f, yu = %f, zl = %f, zu = %f\n", xl, xu , yl, yu, zl, zu);
        printf("this is first mul_inv ... \n");
        mul_inv(r1, r2, r3);
        printf("after first mul_inv: xl = %f, xu = %f, yl = %f, yu = %f, zl = %f, zu = %f\n", xl, xu , yl, yu, zl, zu);
        printf("this is second mul_inv ... \n");
        mul_inv(r1, r3, r2);
        printf("after second mul_inv: xl = %f, xu = %f, yl = %f, yu = %f, zl = %f, zu = %f\n", xl, xu , yl, yu, zl, zu);
        break;
      }
      case MIN: {
        r1.lb() = max(xl, min(yl, zl));
        r1.ub() = min(xu, min(yu, zu));
        r2.lb() = max(yl, xl);
        if(xu < zl) { r2.ub() = min(yu, xu); }
        r3.lb() = max(zl, xl);
        if(xu < yl) { r3.ub() = min(zu, xu); }
        break;
      }
      case MAX: {
        r1.lb() = max(xl, max(yl, zl));
        r1.ub() = min(xu, max(yu, zu));
        r2.ub() = min(yu, xu);
        if(xl > zu) { r2.lb() = max(yl, xl); }
        r3.ub() = min(zu, xu);
        if(xl > yu) { r3.lb() = max(zl, xl); }
        break;
      }
      default: assert(false);
    }
    has_changed |= sub->embed(bytecode.x, r1);
    has_changed |= sub->embed(bytecode.y, r2);
    has_changed |= sub->embed(bytecode.z, r3);
    return has_changed;
  }

#undef xl
#undef xu
#undef yl
#undef yu
#undef zl
#undef zu
#undef INF
#undef MINF

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

private:
  template<class Env, class Allocator2>
  CUDA NI TFormula<Allocator2> deinterpret(bytecode_type bytecode, const Env& env, Allocator2 allocator) const {
    using F = TFormula<Allocator2>;
    auto X = F::make_lvar(bytecode.x.aty(), LVar<Allocator2>(env.name_of(bytecode.x), allocator));
    auto Y = F::make_lvar(bytecode.y.aty(), LVar<Allocator2>(env.name_of(bytecode.y), allocator));
    auto Z = F::make_lvar(bytecode.z.aty(), LVar<Allocator2>(env.name_of(bytecode.z), allocator));
    return F::make_binary(X, EQ, F::make_binary(Y, bytecode.op, Z, aty(), allocator), aty(), allocator);
  }

public:
  template<class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const Env& env, bool remove_entailed, size_t& num_entailed, Allocator2 allocator = Allocator2()) const {
    using F = TFormula<Allocator2>;
    typename F::Sequence seq{allocator};
    seq.push_back(sub->deinterpret(env, allocator));
    for(int i = 0; i < bytecodes->size(); ++i) {
      if(remove_entailed && ask(i)) {
        num_entailed++;
        continue;
      }
      seq.push_back(deinterpret((*bytecodes)[i], env, allocator));
    }
    return F::make_nary(AND, std::move(seq), aty());
  }

  template<class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const Env& env, Allocator2 allocator = Allocator2()) const {
    size_t num_entailed = 0;
    return deinterpret(env, false, num_entailed, allocator);
  }

  template<class I, class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const I& intermediate, const Env& env, Allocator2 allocator = Allocator2()) const {
    using F = TFormula<Allocator2>;
    typename F::Sequence seq{allocator};
    seq.push_back(sub->deinterpret(intermediate.sub_value, env, allocator));
    for(int i = 0; i < intermediate.bytecodes.size(); ++i) {
      seq.push_back(deinterpret(intermediate.bytecodes[i], env, allocator));
    }
    return F::make_nary(AND, std::move(seq), aty());
  }
};

}

#endif
