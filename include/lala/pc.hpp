// Copyright 2021 Pierre Talbot

#ifndef LALA_PC_IPC_HPP
#define LALA_PC_IPC_HPP

#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/root_ptr.hpp"
#include "battery/allocator.hpp"
#include "battery/algorithm.hpp"

#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"
#include "lala/vstore.hpp"

#include "terms.hpp"
#include "formula.hpp"

namespace lala {
template <class A, class Alloc> class PC;
namespace impl {
  template <class>
  struct is_pc_like {
    static constexpr bool value = false;
  };
  template<class A, class Alloc>
  struct is_pc_like<PC<A, Alloc>> {
    static constexpr bool value = true;
  };
}

/** PC is an abstract transformer built on top of an abstract domain `A`.
    It is expected that `A` has a projection function `u = project(x)`.
    We also expect a `tell(x, u, has_changed)` function to join the abstract universe `u` in the domain of the variable `x`.
    An example of abstract domain satisfying these requirements is `VStore<Interval<ZInc>>`. */
template <class A, class Allocator = typename A::allocator_type>
class PC {
public:
  using sub_type = A;
  using universe_type = typename A::universe_type;
  using local_universe_type = typename universe_type::local_type;
  using allocator_type = Allocator;
  using sub_allocator_type = typename A::allocator_type;
  using this_type = PC<sub_type, allocator_type>;

  template <class Alloc>
  struct snapshot_type
  {
    using sub_snap_type = A::template snapshot_type<Alloc>;
    size_t num_props;
    sub_snap_type sub_snap;

    CUDA snapshot_type(size_t num_props, sub_snap_type&& sub_snap)
      : num_props(num_props)
      , sub_snap(std::move(sub_snap))
    {}

    snapshot_type(const snapshot_type<Alloc>&) = default;
    snapshot_type(snapshot_type<Alloc>&&) = default;
    snapshot_type<Alloc>& operator=(snapshot_type<Alloc>&&) = default;
    snapshot_type<Alloc>& operator=(const snapshot_type<Alloc>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc& alloc = Alloc{})
      : num_props(other.num_props)
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
  constexpr static const char* name = "PC";

  template <class A2, class Alloc2>
  friend class PC;

private:
  using formula_type = pc::Formula<A, allocator_type>;
  using term_type = pc::Term<A, allocator_type>;
  template<class Alloc> using term_ptr = battery::unique_ptr<pc::Term<A, Alloc>, Alloc>;

  AType atype;
  sub_ptr sub;
  using props_type = battery::vector<formula_type, allocator_type>;
  battery::root_ptr<props_type, allocator_type> props;

public:
  template <class Alloc>
  using formula_seq = battery::vector<pc::Formula<A, Alloc>, Alloc>;

  template <class Alloc>
  using term_seq = battery::vector<pc::Term<A, Alloc>, Alloc>;

  template <class Alloc, class SubType>
  struct interpreted_type {
    SubType sub_value;
    formula_seq<Alloc> props;

    interpreted_type(interpreted_type&&) = default;
    interpreted_type& operator=(interpreted_type&&) = default;
    interpreted_type(const interpreted_type&) = default;

    CUDA interpreted_type(const SubType& sub_value, const Alloc& alloc = Alloc{})
      : sub_value(sub_value), props(alloc)
    {}

    CUDA interpreted_type(const Alloc& alloc = Alloc{})
      : sub_value(alloc), props(alloc) {}

    template <class InterpretedType>
    CUDA interpreted_type(const InterpretedType& other, const Alloc& alloc = Alloc{})
      : sub_value(other.sub_value, alloc)
      , props(other.props, alloc)
    {}

    template <class Alloc2, class SubType2>
    friend struct interpreted_type;
  };

  template <class Alloc>
  using tell_type = interpreted_type<Alloc, typename sub_type::template tell_type<Alloc>>;

  template <class Alloc>
  using ask_type = interpreted_type<Alloc, typename sub_type::template ask_type<Alloc>>;

  CUDA PC(AType atype, sub_ptr sub, const allocator_type& alloc = allocator_type{})
   : atype(atype), sub(std::move(sub))
   , props(battery::allocate_root<props_type, allocator_type>(alloc, alloc))  {}

  CUDA PC(PC&& other)
    : atype(other.atype)
    , props(std::move(other.props))
    , sub(std::move(other.sub))
  {}

private:
  // When activated (`deps.is_shared_copy()`), we avoid copying the propagators and share them with the ones of the root `other`.
  // This allows to save up memory and to avoid contention on L2 cache among blocks.
  template<class A2, class Alloc2, class... Allocators>
  CUDA static battery::root_ptr<props_type, allocator_type> init_props(const PC<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps) {
    auto alloc = deps.template get_allocator<allocator_type>();
    if constexpr(std::is_same_v<allocator_type, Alloc2>) {
      if(deps.is_shared_copy()) {
        return other.props;
      }
    }
    auto r = battery::allocate_root<props_type, allocator_type>(alloc, *(other.props), alloc);
    return std::move(r);
  }

public:
  template<class A2, class Alloc2, class... Allocators>
  CUDA PC(const PC<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , sub(deps.template clone<A>(other.sub))
   , props(init_props(other, deps))
  {}

  CUDA allocator_type get_allocator() const {
    return props.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return PC{atype, battery::allocate_shared<sub_type>(alloc, sub_type::bot(atype_sub, sub_alloc)), alloc};
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    AType atype_sub = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return PC{atype, battery::allocate_shared<sub_type>(sub_alloc, sub_type::top(atype_sub, sub_alloc)), alloc};
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
  template <bool diagnose, class F, class Alloc>
  CUDA bool interpret_unary(const F& f, term_seq<Alloc>& intermediate, IDiagnostics& diagnostics, term_ptr<Alloc>&& x) const {
    using T = pc::Term<A, Alloc>;
    Alloc alloc = x.get_allocator();
    switch(f.sig()) {
      case NEG: intermediate.push_back(T::make_neg(std::move(x))); break;
      case ABS: intermediate.push_back(T::make_abs(std::move(x))); break;
      default: RETURN_INTERPRETATION_ERROR("Unsupported unary symbol in a term.");
    }
    return true;
  }

  template <bool diagnose, class F, class Alloc>
  CUDA bool interpret_binary(const F& f, term_seq<Alloc>& intermediate, IDiagnostics& diagnostics, term_ptr<Alloc>&& x, term_ptr<Alloc>&& y) const
  {
    using T = pc::Term<A, Alloc>;
    switch(f.sig()) {
      case ADD: intermediate.push_back(T::make_add(std::move(x), std::move(y))); break;
      case SUB: intermediate.push_back(T::make_sub(std::move(x), std::move(y))); break;
      case MUL: intermediate.push_back(T::make_mul(std::move(x), std::move(y))); break;
      case TDIV: intermediate.push_back(T::make_tdiv(std::move(x), std::move(y))); break;
      case FDIV: intermediate.push_back(T::make_fdiv(std::move(x), std::move(y))); break;
      case CDIV: intermediate.push_back(T::make_cdiv(std::move(x), std::move(y))); break;
      case EDIV: intermediate.push_back(T::make_ediv(std::move(x), std::move(y))); break;
      case MIN: intermediate.push_back(T::make_min(std::move(x), std::move(y))); break;
      case MAX: intermediate.push_back(T::make_max(std::move(x), std::move(y))); break;
      default: RETURN_INTERPRETATION_ERROR("Unsupported binary symbol in a term.");
    }
    return true;
  }

  template <bool diagnose, class F, class Alloc>
  CUDA bool interpret_nary(const F& f, term_seq<Alloc>& intermediate, IDiagnostics& diagnostics, term_seq<Alloc>&& operands) const
  {
    using T = pc::Term<A, Alloc>;
    switch(f.sig()) {
      case ADD: intermediate.push_back(T::make_naryadd(std::move(operands))); break;
      case MUL: intermediate.push_back(T::make_narymul(std::move(operands))); break;
      default: RETURN_INTERPRETATION_ERROR("Unsupported nary symbol in a term.");
    }
    return true;
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_sequence(const F& f, Env& env, term_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const
  {
    using T = pc::Term<A, Alloc>;
    Alloc alloc = intermediate.get_allocator();
    term_seq<Alloc> subterms = term_seq<Alloc>(alloc);
    formula_seq<Alloc> subformulas = formula_seq<Alloc>(alloc);
    for(int i = 0; i < f.seq().size(); ++i) {
      // We first try to interpret the formula `f.seq(i)` as a term, if that fails, try as a formula and wrap it in a term.
      if(!interpret_term<kind, diagnose>(f.seq(i), env, subterms, diagnostics)) {
        if(!interpret_formula<kind, diagnose>(f.seq(i), env, subformulas, diagnostics, true)) {
          return false;
        }
        else {
          subterms.push_back(T::make_formula(
            battery::allocate_unique<pc::Formula<A, Alloc>>(alloc, std::move(subformulas.back()))));
        }
      }
    }
    if(subterms.size() == 1) {
      return interpret_unary<diagnose>(f,
        intermediate,
        diagnostics,
        battery::allocate_unique<T>(alloc, std::move(subterms[0])));
    }
    else if(subterms.size() == 2) {
      return interpret_binary<diagnose>(f,
        intermediate,
        diagnostics,
        battery::allocate_unique<T>(alloc, std::move(subterms[0])),
        battery::allocate_unique<T>(alloc, std::move(subterms[1])));
    }
    else {
      return interpret_nary<diagnose>(f,
        intermediate,
        diagnostics,
        std::move(subterms));
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_term(const F& f, Env& env, term_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using T = pc::Term<A, Alloc>;
    using F2 = TFormula<Alloc>;
    if(f.is_variable()) {
      AVar avar;
      if(env.template interpret<diagnose>(f, avar, diagnostics)) {
        intermediate.push_back(T::make_var(avar));
        return true;
      }
      else {
        return false;
      }
    }
    else if(f.is_constant()) {
      auto constant = F2::make_binary(F2::make_avar(AVar{}), EQ, f, UNTYPED, intermediate.get_allocator());
      local_universe_type k{local_universe_type::bot()};
      if(local_universe_type::template interpret<kind, diagnose>(constant, env, k, diagnostics)) {
        intermediate.push_back(T::make_constant(std::move(k)));
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Constant in a term could not be interpreted in the underlying abstract universe.");
      }
    }
    else if(f.is(F::Seq)) {
      return interpret_sequence<kind, diagnose>(f, env, intermediate, diagnostics);
    }
    else {
      RETURN_INTERPRETATION_ERROR("The shape of the formula is not supported in PC, and could not be interpreted as a term.");
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_negation(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, bool neg_context) const {
    auto nf = negate(f);
    if(nf.has_value()) {
      return interpret_formula<kind, diagnose>(*nf, env, intermediate, diagnostics, neg_context);
    }
    else {
      RETURN_INTERPRETATION_ERROR("We must query this formula for disentailement, but we could not compute its negation.");
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc, class Create>
  CUDA bool interpret_binary_logical_connector(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, bool neg_context, Create&& create) const {
    using PF = pc::Formula<A, Alloc>;
    Alloc alloc = intermediate.get_allocator();
    formula_seq<Alloc> operands = formula_seq<Alloc>(alloc);
    if( interpret_formula<kind, diagnose>(f, env, operands, diagnostics, neg_context)
     && interpret_formula<kind, diagnose>(g, env, operands, diagnostics, neg_context))
    {
      intermediate.push_back(create(
        battery::allocate_unique<PF>(alloc, std::move(operands[0])),
        battery::allocate_unique<PF>(alloc, std::move(operands[1]))));
      return true;
    }
    else {
      return false;
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_conjunction(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, bool neg_context) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, neg_context,
      [](auto&& l, auto&& k) { return PF::make_conj(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_disjunction(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_disj(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_biconditional(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_bicond(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_implication(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_imply(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_xor(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_xor(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc, class Create>
  CUDA bool interpret_binary_predicate(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, Create&& create) const
  {
    using PF = pc::Formula<A, Alloc>;
    using T = pc::Term<A, Alloc>;
    Alloc alloc = intermediate.get_allocator();
    term_seq<Alloc> operands = term_seq<Alloc>(alloc);
    if( interpret_term<kind, diagnose>(f, env, operands, diagnostics)
     && interpret_term<kind, diagnose>(g, env, operands, diagnostics))
    {
      intermediate.push_back(create(std::move(operands[0]), std::move(operands[1])));
      return true;
    }
    else {
      return false;
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_equality(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_predicate<kind, diagnose>(f, g, env, intermediate, diagnostics,
      [](auto&& l, auto&& k) { return PF::make_eq(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_disequality(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_predicate<kind, diagnose>(f, g, env, intermediate, diagnostics,
      [](auto&& l, auto&& k) { return PF::make_neq(std::move(l), std::move(k)); });
  }

  template <bool neg, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_literal(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics) const {
    using PF = pc::Formula<A, Alloc>;
    AVar avar{};
    if(env.template interpret<diagnose>(f, avar, diagnostics)) {
      if constexpr(neg) {
        intermediate.push_back(PF::make_nvarlit(avar));
      }
      else {
        intermediate.push_back(PF::make_pvarlit(avar));
      }
      return true;
    }
    return false;
  }

  /** expr != k is transformed into expr < k \/ expr > k.
   * `k` needs to be an integer. */
  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_neq_decomposition(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, bool neg_context) const {
    if(f.sig() == NEQ && f.seq(1).is(F::Z)) {
      using F2 = TFormula<Alloc>;
      Alloc alloc = intermediate.get_allocator();
      return interpret_formula<kind, diagnose>(
        F2::make_binary(
          F2::make_binary(f.seq(0), LT, f.seq(1), f.type(), alloc),
          OR,
          F2::make_binary(f.seq(0), GT, f.seq(1), f.type(), alloc),
          f.type(),
          alloc),
        env,
        intermediate,
        diagnostics,
        neg_context
      );
    }
    else {
      RETURN_INTERPRETATION_ERROR("Unsupported predicate in this abstract domain.");
    }
  }

  /** Given an interval occuring in a set (LogicSet), we decompose it as a formula. */
  template <class F, class Alloc>
  CUDA F itv_to_formula(AType ty, const F& f, const battery::tuple<F, F>& itv, const Alloc& alloc) const {
    using F2 = TFormula<Alloc>;
    if(battery::get<0>(itv) == battery::get<1>(itv)) {
      return F2::make_binary(f, EQ, battery::get<0>(itv), ty, alloc);
    }
    else {
      return
        F2::make_binary(
          F2::make_binary(f, GEQ, battery::get<0>(itv), ty, alloc),
          AND,
          F2::make_binary(f, LEQ, battery::get<1>(itv), ty, alloc),
          ty,
          alloc);
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_in_decomposition(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, bool neg_context = false) const {
    assert(f.seq(1).is(F::S));
    using F2 = TFormula<Alloc>;
    Alloc alloc = intermediate.get_allocator();
    // Decompose IN into disjunction.
    const auto& set = f.seq(1).s();
    if(set.size() == 1) {
      return interpret_formula<kind, diagnose>(
        itv_to_formula(f.type(), f.seq(0), set[0], alloc),
        env, intermediate, diagnostics, neg_context);
    }
    else {
      typename F2::Sequence disjunction = typename F2::Sequence(alloc);
      disjunction.reserve(set.size());
      for(size_t i = 0; i < set.size(); ++i) {
        disjunction.push_back(itv_to_formula(f.type(), f.seq(0), set[i], alloc));
      }
      return interpret_formula<kind, diagnose>(
        F2::make_nary(OR, std::move(disjunction), f.type()),
        env, intermediate, diagnostics, neg_context);
    }
  }

  template <class F>
  CUDA F binarize(const F& f, size_t i) const {
    assert(f.is(F::Seq) && f.seq().size() >= 2);
    if(i + 2 == f.seq().size()) {
      return F::make_binary(f.seq(i), f.sig(), f.seq(i+1), f.type(), f.seq().get_allocator(), false);
    }
    else {
      return F::make_binary(f.seq(i), f.sig(), binarize(f, i+1), f.type(), f.seq().get_allocator(), false);
    }
  }

  /** We interpret the formula `f` in the value `intermediate`, note that we only grow `intermediate` by 0 (if interpretation failed) or 1 (if it succeeds).
   * It is convenient to use a vector because it carries the allocator, and it is the type of the `props` component of the tell/ask type.
   */
  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_formula(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics& diagnostics, bool neg_context = false) const {
    using PF = pc::Formula<A, Alloc>;
    using F2 = TFormula<Alloc>;
    if(f.type() != aty() && !f.is_untyped() && !f.is_variable()) {
      RETURN_INTERPRETATION_ERROR("The type of the formula does not match the type of this abstract domain.");
    }
    Alloc alloc = intermediate.get_allocator();
    if(f.is_false()) {
      intermediate.push_back(PF::make_false());
      return true;
    }
    else if(f.is_true()) {
      intermediate.push_back(PF::make_true());
      return true;
    }
    else if(f.is(F::Seq) && f.sig() == IN) {
      return interpret_in_decomposition<kind, diagnose>(f, env, intermediate, diagnostics, neg_context);
    }
    else if(f.is(F::Seq) && f.seq().size() > 2 && (f.sig() == AND || f.sig() == OR || f.sig() == EQUIV)) {
      return interpret_formula<kind, diagnose>(binarize(f,0), env, intermediate, diagnostics, neg_context);
    }
    else if(f.is_binary()) {
      Sig sig = f.sig();
      switch(sig) {
        case AND: return interpret_conjunction<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics, neg_context);
        case OR:  return interpret_disjunction<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
        case EQUIV: return interpret_biconditional<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
        case IMPLY: return interpret_implication<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
        case XOR: return interpret_xor<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
        case EQ: {
          // Whenever an operand of `=` is a formula with logical connectors, we interpret `=` as `<=>`.
          if(f.seq(0).is_logical() || f.seq(1).is_logical()) {
            return interpret_biconditional<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
          }
          else {
            return interpret_equality<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
          }
        }
        case NEQ: return interpret_disequality<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
        // Expect the shape of the constraint to be `T <op> u`.
        // If `T` is a variable (`x <op> u`), then it is interpreted in the underlying universe.
        default:
          auto fn = move_constants_on_rhs(f);
          auto fu = F2::make_binary(F2::make_avar(AVar{}), fn.sig(), fn.seq(1), fn.type(), alloc);
          local_universe_type u{local_universe_type::bot()};
          // We first try to interpret the right-hand side of the formula.
          if(!local_universe_type::template interpret<kind, diagnose>(fu, env, u, diagnostics)) {
            // Sometimes the underlying universe cannot interpret disequality, so we decompose it.
            if(fn.sig() == NEQ) {
              return interpret_neq_decomposition<kind, diagnose>(fn, env, intermediate, diagnostics, neg_context);
            }
            else {
              RETURN_INTERPRETATION_ERROR("We cannot interpret the constant on the RHS of the formula in the underlying abstract universe.");
            }
          }
          // We continue with the interpretation of the left-hand side of the formula.
          else {
            term_seq<Alloc> terms = term_seq<Alloc>(alloc);
            if(!interpret_term<kind, diagnose>(fn.seq(0), env, terms, diagnostics)) {
              RETURN_INTERPRETATION_ERROR("We cannot interpret the term on the LHS of the formula in PC.");
            }
            else {
              // In a context where the formula propagator can be asked for its negation, we must interpret the negation of the formula as well.
              if(neg_context) {
                auto nf_ = negate(fn);
                if(!nf_.has_value()) {
                  RETURN_INTERPRETATION_ERROR("We must query this formula for disentailement, but we could not compute its negation.");
                }
                else {
                  formula_seq<Alloc> nf;
                  if(!interpret_formula<kind, diagnose>(*nf_, env, nf, diagnostics)) {
                    RETURN_INTERPRETATION_ERROR("We must query this formula for disentailement, but we could not interpret its negation.");
                  }
                  else {
                    intermediate.push_back(PF::make_nlop(std::move(terms.back()), std::move(u),
                      battery::allocate_unique<PF>(alloc, std::move(nf.back()))));
                    return true;
                  }
                }
              }
              else {
                intermediate.push_back(PF::make_plop(std::move(terms.back()), std::move(u)));
                return true;
              }
            }
          }
      }
    }
    // Negative literal
    else if(f.is(F::Seq) && f.seq().size() == 1 && f.sig() == NOT &&
      f.seq(0).is_variable())
    {
      return interpret_literal<true, diagnose>(f.seq(0), env, intermediate, diagnostics);
    }
    // Positive literal
    else if(f.is_variable()) {
      return interpret_literal<false, diagnose>(f, env, intermediate, diagnostics);
    }
    // Logical negation
    else if(f.is(F::Seq) && f.seq().size() == 1 && f.sig() == NOT) {
      return interpret_negation<kind, diagnose>(f.seq(0), env, intermediate, diagnostics, neg_context);
    }
    RETURN_INTERPRETATION_ERROR("The shape of this formula is not supported.");
  }

public:
  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f, Env& env, I& intermediate, IDiagnostics& diagnostics) const {
    size_t error_context = 0;
    if constexpr(diagnose) {
      diagnostics.add_suberror(IDiagnostics(false, name, "Uninterpretable formula in both PC and its sub-domain.", f));
      error_context = diagnostics.num_suberrors();
    }
    bool res = false;
    AType current = f.type();
    const_cast<F&>(f).type_as(sub->aty()); // We restore the type after the call to sub->interpret.
    if(sub->template interpret<kind, diagnose>(f, env, intermediate.sub_value, diagnostics)) {
      // A successful interpretation in the sub-domain does not mean it is interpreted exactly.
      // Sometimes, we can improve the precision by interpreting it in PC.
      // This is the case of `x in S` predicate for sub-domain that do not preserve meet.
      if(!(f.is_binary() && f.sig() == IN && f.seq(0).is_variable() && f.seq(1).is(F::S) && f.seq(1).s().size() > 1)) {
        res = true; // it is not a formula `x in S`.
      }
      else {
        res = universe_type::preserve_meet; // it is `x in S` but it preserves meet.
      }
    }
    const_cast<F&>(f).type_as(current);
    if(!res) {
      res = interpret_formula<kind, diagnose>(f, env, intermediate.props, diagnostics);
    }
    if constexpr(diagnose) {
      diagnostics.merge(res, error_context);
    }
    return res;
  }

  /** PC expects a non-conjunctive formula \f$ c \f$ which can either be interpreted in the sub-domain `A` or in the current domain.
  */
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics& diagnostics) const {
    return interpret<IKind::TELL, diagnose>(f, env, tell, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc2>& ask, IDiagnostics& diagnostics) const {
    return interpret<IKind::ASK, diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
  }

  /** Note that we cannot add propagators in parallel (but modifying the underlying domain is ok).
      This is a current limitation that we should fix later on.
      Notes for later:
        * To implement "telling of propagators", we would need to check if a propagator has already been added or not (for idempotency).
        * 1. Walk through the existing propagators to check which ones are already in.
        * 2. If a propagator has the same shape but different constant `U`, join them in place.  */
  template <class Alloc2, class Mem>
  CUDA this_type& tell(const tell_type<Alloc2>& t, BInc<Mem>& has_changed) {
    sub->tell(t.sub_value, has_changed);
    if(t.props.size() > 0) {
      has_changed.tell_top();
      auto& props2 = *props;
      size_t n = props2.size();
      props2.reserve(n + t.props.size());
      for(int i = 0; i < t.props.size(); ++i) {
        props2.push_back(formula_type(t.props[i], get_allocator()));
        props2[n + i].preprocess(*sub, has_changed);
      }
      battery::vector<size_t> lengths(props2.size());
      for(int i = 0; i < props2.size(); ++i) {
        lengths[i] = props2[i].length();
      }
      battery::sorti(props2,
        [&](int i, int j) {
          return props2[i].kind() < props2[j].kind() || (props2[i].kind() == props2[j].kind() && lengths[i] < lengths[j]);
        });
    }
    return *this;
  }

  template <class Alloc2>
  CUDA this_type& tell(const tell_type<Alloc2>& t) {
    local::BInc has_changed;
    return tell(t, has_changed);
  }

  CUDA this_type& tell(AVar x, const universe_type& dom) {
    sub->tell(x, dom);
    return *this;
  }

  template <class Mem>
  CUDA this_type& tell(AVar x, const universe_type& dom, BInc<Mem>& has_changed) {
    sub->tell(x, dom, has_changed);
    return *this;
  }

  template <class Alloc2>
  CUDA local::BInc ask(const ask_type<Alloc2>& t) const {
    for(int i = 0; i < t.props.size(); ++i) {
      if(!t.props[i].ask(*sub)) {
        return false;
      }
    }
    return sub->ask(t.sub_value);
  }

  CUDA size_t num_refinements() const {
    return props->size();
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    assert(i < num_refinements());
    if(is_top()) { return; }
    (*props)[i].refine(*sub, has_changed);
  }

  // Functions forwarded to the sub-domain `A`.

  /** `true` if the underlying abstract element is top, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return sub->is_top();
  }

  /** `true` if the underlying abstract element is bot and there is no refinement function, `false` otherwise. */
  CUDA local::BDec is_bot() const {
    return sub->is_bot() && props->size() == 0;
  }

  CUDA auto /* universe_type or const universe_type& */ operator[](int x) const {
    return (*sub)[x];
  }

  CUDA auto /* universe_type or const universe_type& */ project(AVar x) const {
    return sub->project(x);
  }

  CUDA size_t vars() const {
    return sub->vars();
  }

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>(props->size(), sub->snapshot(alloc));
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    size_t n = props->size();
    for(size_t i = snap.num_props; i < n; ++i) {
      props->pop_back();
    }
    sub->restore(snap.sub_snap);
  }

  /** An abstract element is extractable when it is not equal to top, if all propagators are entailed and if the underlying abstract element is extractable. */
  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    if(is_top()) {
      return false;
    }
    for(int i = 0; i < props->size(); ++i) {
      if(!(*props)[i].ask(*sub)) {
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
    if constexpr(impl::is_pc_like<B>::value) {
      sub->extract(*ua.sub);
    }
    else {
      sub->extract(ua);
    }
  }

  template<class Env>
  CUDA NI TFormula<typename Env::allocator_type> deinterpret(const Env& env) const {
    using F = TFormula<typename Env::allocator_type>;
    F sub_f = sub->deinterpret(env);
    typename F::Sequence seq{env.get_allocator()};
    if(sub_f.is(F::Seq) && sub_f.sig() == AND) {
      for(int i = 0; i < sub_f.seq().size(); ++i) {
        seq.push_back(sub_f.seq(i));
      }
    }
    else {
      seq.push_back(sub_f);
    }
    for(int i = 0; i < props->size(); ++i) {
      seq.push_back((*props)[i].deinterpret(env.get_allocator(), aty()));
      map_avar_to_lvar(seq.back(), env);
    }
    return F::make_nary(AND, std::move(seq), aty());
  }
};

}

#endif
