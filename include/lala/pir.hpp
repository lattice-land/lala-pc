// Copyright 2024 Pierre Talbot

#ifndef LALA_PC_PIR_HPP
#define LALA_PC_PIR_HPP

#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/root_ptr.hpp"
#include "battery/allocator.hpp"
#include "battery/algorithm.hpp"

#include "lala/logic/logic.hpp"
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
    size_t num_bytecodes;
    sub_snap_type sub_snap;

    CUDA snapshot_type(size_t num_bytecodes, sub_snap_type&& sub_snap)
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

private:
  AType atype;
  sub_ptr sub;

  const local_universe_type ZERO;
  const local_universe_type ONE;

  static_assert(sizeof(int) == sizeof(AVar), "The size of AVar must be equal to the size of an int.");
  static_assert(sizeof(int) == sizeof(Sig), "The size of Sig must be equal to the size of an int.");
  union bytecode_type
  {
    // This represents the constraints `X = Y [op] Z`.
    struct {
      Sig op;
      AVar x;
      AVar y;
      AVar z;
    };
    int4 code;
  };
  using bytecodes_type = battery::vector<bytecode_type, allocator_type>;
  using bytecodes_ptr = battery::root_ptr<battery::vector<bytecode_type, allocator_type>>;

  /** We represent the constraints X = Y [op] Z in a struct of array manner. */
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
        return other.bytecodes;
      }
    }
    auto r = battery::allocate_root<bytecodes_type, allocator_type>(alloc, *(other.bytecodes), alloc);
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
      // Expect constraint of the form X = Y <OP> Z.
      if(sig == EQ && f.seq(1).is_binary()) {
        auto& X = f.seq(0);
        auto& Y = f.seq(1).seq(0);
        auto& Z = f.seq(1).seq(1);
        Bytecode bytecode;
        bytecode.op = f.seq(1).sig();
        if(X.is_variable() && Y.is_variable() && Z.is_variable() &&
          (op == ADD || op == MUL || op == SUB || op == EDIV || op == EMOD || op == MIN || op == MAX)
        {
          if( env.template interpret<diagnose>(X, bytecode.x, diagnostics)
           && env.template interpret<diagnose>(Y, bytecode.y, diagnostics)
           && env.template interpret<diagnose>(Z, bytecode.z, diagnostics))
          {
            intermediate.bytecodes.push_back(bytecode);
            return true;
          }
          return false;
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
    res = sub->template interpret<kind, diagnose>(f, env, intermediate.sub_value, diagnostics);
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
      }
      /** This is sorting the constraints `X = Y <op> Z` according to <OP>. */
      battery::sorti(*bytecodes,
        [&](int i, int j) { return (*bytecodes)[i].op < (*bytecodes)[j].op; });
      has_changed = true;
    }
    return has_changed;
  }

  CUDA bool embed(AVar x, const universe_type& dom) {
    return sub->embed(x, dom);
  }

  CUDA local::B ask(size_t i) const {

  }

  CUDA size_t num_deductions() const {
    return bytecodes->size();
  }

public:
  CUDA local::B deduce(size_t i) {
    assert(i < num_deductions());

    // Vectorize load (int4).
    Bytecode bytecode = (*bytecodes)[i];

    local::B has_changed = false;

    // We load the variables.
    local_universe_type r1;
    local_universe_type r2((*sub)[bytecode.y]);
    local_universe_type r3((*sub)[bytecode.z]);

    // Reified constraint: X = (Y = Z) and X = (Y <= Z).
    if(bytecode.op == EQ || bytecode.op == LEQ) {
      r1 = (*sub)[bytecode.x];
      // Y [op] Z
      if(r1 >= ONE) {
        if(bytecode.op == EQ) {
          has_changed |= sub->embed(bytecode.y, r3);
          has_changed |= sub->embed(bytecode.z, r2);
        }
        else {
          assert(bytecode.op == LEQ);
          r3.join_lb(LB::top());
          r2.join_ub(UB::top());
          has_changed |= sub->embed(bytecode.y, r3);
          has_changed |= sub->embed(bytecode.z, r2);
        }
      }
      // not (Y [op] Z)
      else if(r1 >= ZERO) {
        if(bytecode.op == EQ) {
          if(r2.lb().value() == r2.ub().value()) {
            r1 = r3;
            r1.meet_lb(LB::prev(r2.lb()));
            r3.meet_ub(UB::prev(r2.ub()));
            has_changed |= sub->embed(bytecode.y, fjoin(r1,r3));
          }
          else if(r3.lb().value() == r3.ub().value()) {
            r1 = r2;
            r1.meet_lb(LB::prev(r3.lb()));
            r2.meet_ub(UB::prev(r3.ub()));
            has_changed |= sub->embed(bytecode.z, fjoin(r1,r2));
          }
        }
        else {
          assert(bytecode.op == LEQ);
          r3.meet_lb(LB::prev(r3.lb()));
          r2.meet_ub(UB::prev(r2.ub()));
          has_changed |= sub->embed(bytecode.y, r3);
          has_changed |= sub->embed(bytecode.z, r2);
        }
      }
      // X <- 1
      else if(r2.ub().value() <= r3.lb().value() && (bytecode.op == LEQ || r2.lb().value() == r3.ub().value())) {
        has_changed |= sub->embed(bytecode.x, ONE);
      }
      // X <- 0
      else if(r2.lb().value() > r3.ub().value() || (bytecode.op == EQ && r2.ub().value() < r3.lb().value())) {
        has_changed |= sub->embed(bytecode.x, ZERO);
      }
    }
    // Arithmetic constraint: X = Y + Z, X = Y - Z, ...
    else {
      // X <- Y [op] Z
      r1.project(bytecode.op, r2, r3);
      sub->embed(bytecode.x, r1);

      // Y <- X <left residual> Z
      r1 = (*sub)[bytecode.x];
      switch(bytecode.op) {
        case ADD: GroupAdd<local_universe_type>::left_residual(r1, r3, r2); break;
        case SUB: GroupSub<local_universe_type>::left_residual(r1, r3, r2); break;
        case MUL: GroupMul<local_universe_type, EDIV>::left_residual(r1, r3, r2); break;
        case EDIV: GroupDiv<local_universe_type, EDIV>::left_residual(r1, r3, r2); break;
        case MIN: GroupMinMax<local_universe_type, MIN>::left_residual(r1, r3, r2); break;
        case MAX: GroupMinMax<local_universe_type, MAX>::left_residual(r1, r3, r2); break;
        default: assert(false);
      }
      has_changed |= sub->embed(bytecode.y, r2);

      // Z <- X <right residual> Y
      r2 = (*sub)[bytecode.y];
      r3.join_top();
      switch(bytecode.op) {
        case ADD: GroupAdd<local_universe_type>::right_residual(r1, r2, r3); break;
        case SUB: GroupSub<local_universe_type>::right_residual(r1, r2, r3); break;
        case MUL: GroupMul<local_universe_type, EDIV>::right_residual(r1, r2, r3); break;
        case EDIV: GroupDiv<local_universe_type, EDIV>::right_residual(r1, r2, r3); break;
        case MIN: GroupMinMax<local_universe_type, MIN>::right_residual(r1, r2, r3); break;
        case MAX: GroupMinMax<local_universe_type, MAX>::right_residual(r1, r2, r3); break;
        default: assert(false);
      }
      has_changed |= sub->embed(bytecode.z, r3);
    }
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

  CUDA size_t vars() const {
    return sub->vars();
  }

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>(bytecodes->size(), sub->snapshot(alloc));
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    size_t n = bytecodes->size();
    for(size_t i = snap.num_bytecodes; i < n; ++i) {
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
  CUDA NI TFormula<Allocator2> deinterpret(Bytecode bytecode, const Env& env, Allocator2 allocator) const {
    auto X = F::make_lvar(bytecode.x.aty(), LVar<Allocator>(env.name_of(bytecode.x), allocator));
    auto Y = F::make_lvar(bytecode.y.aty(), LVar<Allocator>(env.name_of(bytecode.y), allocator));
    auto Z = F::make_lvar(bytecode.z.aty(), LVar<Allocator>(env.name_of(bytecode.z), allocator));
    return F::make_binary(X, EQ, F::make_binary(Y, bytecode.op, Z, allocator), aty(), allocator);
  }
public:

  template<class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const Env& env, Allocator2 allocator = Allocator2()) const {
    using F = TFormula<Allocator2>;
    typename F::Sequence seq{allocator};
    seq.push_back(sub->deinterpret(env, allocator));
    for(int i = 0; i < bytecodes->size(); ++i) {
      seq.push_back(deinterpret((*bytecodes)[i], env, allocator));
    }
    return F::make_nary(AND, std::move(seq), aty());
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
