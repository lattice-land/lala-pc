// Copyright 2021 Pierre Talbot

#ifndef LALA_PC_IPC_HPP
#define LALA_PC_IPC_HPP

#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/allocator.hpp"

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
  using allocator_type = Allocator;
  using sub_allocator_type = typename A::allocator_type;
  using this_type = PC<sub_type, allocator_type>;

  template <class Alloc2>
  struct snapshot_type
  {
    using sub_snap_type = A::template snapshot_type<Alloc2>;
    size_t num_props;
    sub_snap_type sub_snap;

    CUDA snapshot_type(size_t num_props, sub_snap_type&& sub_snap)
      : num_props(num_props)
      , sub_snap(std::move(sub_snap))
    {}

    snapshot_type(const snapshot_type<Alloc2>&) = default;
    snapshot_type(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(snapshot_type<Alloc2>&&) = default;
    snapshot_type<Alloc2>& operator=(const snapshot_type<Alloc2>&) = default;

    template <class SnapshotType>
    CUDA snapshot_type(const SnapshotType& other, const Alloc2& alloc = Alloc2())
      : num_props(other.num_props)
      , sub_snap(other.sub_snap, alloc)
    {}
  };

  using sub_ptr = abstract_ptr<sub_type>;

  constexpr static const char* name = "PC";

  template <class A2, class Alloc2>
  friend class PC;

private:
  using formula_type = pc::Formula<A, allocator_type>;
  using term_type = pc::Term<A, allocator_type>;
  template<class Alloc> using term_ptr = battery::unique_ptr<pc::Term<A, Alloc>, Alloc>;

  AType atype;
  sub_ptr sub;
  battery::vector<formula_type, allocator_type> props;

public:
  template <class Alloc2>
  using formula_seq = battery::vector<pc::Formula<A, Alloc2>, Alloc2>;

  template <class Alloc2, class SubType>
  struct interpreted_type {
    SubType sub_value;
    formula_seq props;

    interpreted_type(interpreted_type&&) = default;
    interpreted_type& operator=(interpreted_type&&) = default;
    interpreted_type(const interpreted_type&) = default;

    CUDA interpreted_type(const SubType& sub_value, const Alloc2& alloc = Alloc2())
      : sub_value(sub_value), props(alloc)
    {}

    CUDA interpreted_type(const Alloc2& alloc = Alloc2())
      : sub_value(alloc), props(alloc) {}

    template <class InterpretedType>
    CUDA interpreted_type(const InterpretedType& other, const Alloc2& alloc = Alloc2())
      : sub_value(other.sub_value, alloc)
      , props(other.props, alloc)
    {}

    template <class Alloc3, class SubType2>
    friend struct interpreted_type;
  };

  template <class Alloc2>
  using tell_type = interpreted_type<Alloc2, typename sub_type::template tell_type<Alloc2>>;

  template <class Alloc2>
  using ask_type = interpreted_type<Alloc2, typename sub_type::template ask_type<Alloc2>>;

public:
  CUDA PC(AType atype, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : atype(atype), sub(std::move(sub)), props(alloc)  {}

  CUDA PC(PC&& other)
    : atype(other.atype)
    , props(std::move(other.props))
    , sub(std::move(other.sub))
  {}

  template<class A2, class Alloc2, class... Allocators>
  CUDA NI PC(const PC<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
   : atype(other.atype)
   , sub(deps.template clone<A>(other.sub))
   , props(other.props, deps.template get_allocator<allocator_type>())
  {}

  CUDA allocator_type get_allocator() const {
    return props.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  CUDA static this_type bot(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return PC(atype, battery::allocate_shared<sub_type>(alloc, sub_type::bot(UNTYPED, sub_alloc)), alloc);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED,
    const allocator_type& alloc = allocator_type(),
    const sub_allocator_type& sub_alloc = sub_allocator_type())
  {
    return PC(atype, battery::allocate_shared<sub_type>(sub_alloc, sub_type::top(UNTYPED, sub_alloc)), alloc);
  }

private:
  template<class Alloc, class F>
  CUDA NI static tresult<Alloc, F> interpret_unary(const F& f, term_ptr<Alloc>&& a) {
    using T = pc::Term<A, Alloc>;
    Alloc alloc = a.get_allocator();
    switch(f.sig()) {
      case NEG: return T::make_neg(std::move(a));
      case ABS: return T::make_abs(std::move(a));
      default: return tresult<Alloc, F>(IError<F>(true, name, "Unsupported unary symbol.", f));
    }
  }

  template<class Alloc, class F>
  CUDA NI static tresult<Alloc, F> interpret_binary(const F& f, term_ptr<Alloc>&& x, term_ptr<Alloc>&& y)
  {
    using T = pc::Term<A, Alloc>;
    switch(f.sig()) {
      case ADD: return T::make_add(std::move(x), std::move(y));
      case SUB: return T::make_sub(std::move(x), std::move(y));
      case MUL: return T::make_mul(std::move(x), std::move(y));
      case TDIV: return T::make_tdiv(std::move(x), std::move(y));
      case FDIV: return T::make_fdiv(std::move(x), std::move(y));
      case CDIV: return T::make_cdiv(std::move(x), std::move(y));
      case EDIV: return T::make_ediv(std::move(x), std::move(y));
      case MIN: return T::make_min(std::move(x), std::move(y));
      case MAX: return T::make_max(std::move(x), std::move(y));
      default: return tresult<Alloc, F>(IError<F>(true, name, "Unsupported binary symbol.", f));
    }
  }

  template<class Alloc, class F>
  CUDA NI static tresult<Alloc, F> interpret_nary(const F& f, battery::vector<pc::Term<A, Alloc>, Alloc>&& subterms)
  {
    using T = pc::Term<A, Alloc>;
    switch(f.sig()) {
      case ADD: return T::make_naryadd(std::move(subterms));
      case MUL: return T::make_narymul(std::move(subterms));
      default: return tresult<Alloc, F>(IError<F>(true, name, "Unsupported nary symbol.", f));
    }
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA NI tresult<Alloc, F> interpret_sequence(const F& f, Env& env)
  {
    using T = pc::Term<A, Alloc>;
    Alloc alloc = env.get_allocator();
    battery::vector<T, Alloc> subterms(alloc);
    subterms.reserve(f.seq().size());
    for(int i = 0; i < f.seq().size(); ++i) {
      auto t = interpret_term<is_tell>(f.seq(i), env);
      if(!t.has_value()) {
        auto p = interpret_formula<is_tell>(f.seq(i), env, true);
        if(!p.has_value()) {
          return std::move(t.join_errors(std::move(p)));
        }
        t = tresult<Alloc, F>(T::make_formula(
          battery::allocate_unique<pc::Formula<A, Alloc>>(alloc, std::move(p.value()))));
        t.join_warnings(std::move(p));
      }
      subterms.push_back(std::move(t.value()));
    }
    if(subterms.size() == 1) {
      return interpret_unary(f,
        battery::allocate_unique<T>(alloc, std::move(subterms[0])));
    }
    else if(subterms.size() == 2) {
      return interpret_binary(f,
        battery::allocate_unique<T>(alloc, std::move(subterms[0])),
        battery::allocate_unique<T>(alloc, std::move(subterms[1])));
    }
    else {
      return interpret_nary(f, std::move(subterms));
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_term(const F& f, Env& env, pc::Term<A, Alloc>& term, IDiagnostics<F>& diagnostics, const Alloc& alloc, bool neg_context) {
    using T = pc::Term<A, Alloc>;
    using F2 = TFormula<Alloc>;
    if(f.is_variable()) {
      AVar avar;
      if(env.template interpret<diagnose>(f, avar, diagnostics)) {
        term = T::make_var(avar);
        return true;
      }
      else {
        return false;
      }
    }
    else if(f.is_constant()) {
      auto constant = F2::make_binary(F2::make_avar(AVar{}), EQ, f, UNTYPED, alloc);
      universe_type k{universe_type::bot()};
      if(universe_type::template interpret<kind, diagnose>(constant, env, k, diagnostics)) {
        term = T::make_constant(std::move(k));
        return true;
      }
      else {
        RETURN_INTERPRETATION_ERROR("Constant in a term could not be interpreted in the underlying abstract universe.");
      }
    }
    else if(f.is(F::Seq)) {
      return interpret_sequence<kind, diagnose>(f, env, term, diagnostics, alloc);
    }
    else {
      RETURN_INTERPRETATION_ERROR("The shape of the formula is not supported in PC, and could not be interpreted as a term.");
    }
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA NI fresult<Alloc, F> interpret_negation(const F& f, Env& env, bool neg_context) {
    auto nf = negate(f);
    if(nf.has_value()) {
      return interpret_formula<is_tell>(*nf, env, neg_context);
    }
    else {
      return fresult<Alloc, F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not compute its negation.", f));
    }
  }


  template <IKind kind, bool diagnose, class Create, class F, class Env, class Alloc>
  CUDA bool interpret_binary_logical_connector(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics, bool neg_context, Create&& create)  {
    using PF = pc::Formula<A, Alloc>;
    Alloc alloc = seq.get_allocator();

KEEP GOING HERE!

    auto l = interpret_formula<kind, diagnose>(f, env, intermediate, diagnostics, neg_context);
    if(l.has_value()) {
      auto k = interpret_formula<kind, diagnose>(g, env, intermediate, diagnostics, neg_context);
      if(k.has_value()) {
        return std::move(fresult<Alloc, F>(create(
            battery::allocate_unique<PF>(alloc, std::move(l.value())),
            battery::allocate_unique<PF>(alloc, std::move(k.value())))
          )
          .join_warnings(std::move(l))
          .join_warnings(std::move(k)));
        }
      else {
        return std::move(k);
      }
    }
    else {
      return std::move(l);
    }
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_conjunction(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, neg_context,
      [](auto&& l, auto&& k) { return PF::make_conj(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_disjunction(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_disj(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_biconditional(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_bicond(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_implication(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_imply(std::move(l), std::move(k)); });
  }

  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_xor(const F& f, const F& g, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<kind, diagnose>(f, g, env, intermediate, diagnostics, true,
      [](auto&& l, auto&& k) { return PF::make_xor(std::move(l), std::move(k)); });
  }

  template <bool neg, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_literal(const F& f, Env& env, formula_seq<Alloc>& seq, IDiagnostics<F>& diagnostics) {
    using PF = pc::Formula<A, Alloc>;
    AVar avar{};
    if(env.template interpret<diagnose>(f, avar, diagnostics)) {
      if constexpr(neg) {
        seq.push_back(PF::make_nvarlit(avar));
      }
      else {
        seq.push_back(PF::make_pvarlit(avar));
      }
      return true;
    }
    return false;
  }

  /** expr != k is transformed into expr < k \/ expr > k.
   * `k` needs to be an integer. */
  template <IKind kind, bool diagnose, class F, class Env, class Alloc>
  CUDA bool interpret_neq_decomposition(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics, bool neg_context) {
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
  CUDA F itv_to_formula(AType ty, const F& f, const battery::tuple<F, F>& itv, const Alloc& alloc) {
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
  CUDA bool interpret_in_decomposition(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics, bool neg_context = false) {
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
      typename F2::Sequence disjunction{set.size(), alloc};
      for(size_t i = 0; i < set.size(); ++i) {
        disjunction[i] = itv_to_formula(f.seq(0), set[i], alloc);
      }
      return interpret_formula<kind, diagnose>(
        F2::make_nary(OR, std::move(disjunction), f.type()),
        env, intermediate, diagnostics, neg_context);
    }
  }

  template <class F>
  CUDA F binarize(const F& f, size_t i) {
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
  CUDA bool interpret_formula(const F& f, Env& env, formula_seq<Alloc>& intermediate, IDiagnostics<F>& diagnostics, bool neg_context = false) {
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
        case EQ: // Whenever an operand of `=` is a formula with logical connectors, we interpret `=` as `<=>`.
          if(f.seq(0).is_logical() || f.seq(1).is_logical()) {
            return interpret_biconditional<kind, diagnose>(f.seq(0), f.seq(1), env, intermediate, diagnostics);
          }
        // Expect the shape of the constraint to be `T <op> u`.
        // If `T` is a variable (`x <op> u`), then it is interpreted in the underlying universe.
        default:
          auto fn = move_constants_on_rhs(f);
          auto fu = F2::make_binary(F2::make_avar(AVar{}), fn.sig(), fn.seq(1), fn.type(), alloc);
          universe_type u{universe_type::bot()};
          // We first try to interpret the right-hand side of the formula.
          if(!universe_type::template interpret<kind, diagnose>(fu, env, u, diagnostics)) {
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
            pc::Term<A, Alloc> term;
            if(!interpret_term<kind, diagnose>(fn.seq(0), env, term, diagnostics)) {
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
                    intermediate.push_back(PF::make_nlop(std::move(term), std::move(u),
                      battery::allocate_unique<PF>(alloc, std::move(nf.back()))));
                    return true;
                  }
                }
              }
              else {
                intermediate.push_back(PF::make_plop(std::move(term), std::move(u)));
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
      return interpret_negation<kind, diagnostics>(f, env, intermediate, diagnostics, neg_context);
    }
    else {
      RETURN_INTERPRETATION_ERROR("The shape of this formula is not supported.");
    }
  }

public:
  template <IKind kind, bool diagnose = false, class F, class Env, class I>
  CUDA NI bool interpret(const F& f, Env& env, I& intermediate, IDiagnostics<F>& diagnostics) const {
    CALL_WITH_ERROR_CONTEXT(
      "Uninterpretable formula in both PC and its sub-domain.",
      (sub->template interpret<kind, diagnose>(f, env, intermediate.sub_value, diagnostics) ||
      interpret_formula<kind, diagnose>(f, env, intermediate.props, diagnostics)));
  }

  /** PC expects a non-conjunctive formula \f$ c \f$ which can either be interpreted in the sub-domain `A` or in the current domain.
  */
  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_tell(const F& f, Env& env, tell_type<Alloc2>& tell, IDiagnostics<F>& diagnostics) const {
    return interpret<IKind::TELL, diagnose>(f, env, tell, diagnostics);
  }

  template <bool diagnose = false, class F, class Env, class Alloc2>
  CUDA NI bool interpret_ask(const F& f, const Env& env, ask_type<Alloc2>& ask, IDiagnostics<F>& diagnostics) const {
    return const_cast<this_type*>(this)->interpret<IKind::ASK, diagnose>(f, const_cast<Env&>(env), ask, diagnostics);
  }

  /** Note that we cannot add propagators in parallel (but modifying the underlying domain is ok).
      This is a current limitation that we should fix later on.
      Notes for later:
        * To implement "telling of propagators", we would need to check if a propagator has already been added or not (for idempotency).
        * 1. Walk through the existing propagators to check which ones are already in.
        * 2. If a propagator has the same shape but different constant `U`, join them in place.  */
  template <class Alloc2, class Mem>
  CUDA this_type& tell(const tell_type<Alloc2>& t, BInc<Mem>& has_changed) {
    for(int i = 0; i < t.sub_tells.size(); ++i) {
      sub->tell(t.sub_tells[i], has_changed);
    }
    if(t.props.size() > 0) {
      has_changed.tell_top();
    }
    size_t n = props.size();
    props.reserve(n + t.props.size());
    local::BInc has_changed2;
    for(int i = 0; i < t.props.size(); ++i) {
      props.push_back(formula_type(t.props[i], get_allocator()));
      props[n + i].preprocess(*sub, has_changed2);
    }
    has_changed.tell(has_changed2);
    return *this;
  }

  template <class Alloc2>
  CUDA this_type& tell(const tell_type<Alloc2>& t) {
    local::BInc has_changed;
    return tell(t, has_changed);
  }

  template <class Alloc2>
  CUDA local::BInc ask(const tell_type<Alloc2>& t) const {
    for(int i = 0; i < t.props.size(); ++i) {
      if(!t.props[i].ask(*sub)) {
        return false;
      }
    }
    for(int i = 0; i < t.sub_tells.size(); ++i) {
      if(!sub->ask(t.sub_tells[i])) {
        return false;
      }
    }
    return true;
  }

  CUDA size_t num_refinements() const {
    return props.size();
  }

  template <class Mem>
  CUDA void refine(size_t i, BInc<Mem>& has_changed) {
    assert(i < num_refinements());
    if(is_top()) { return; }
    local::BInc has_changed2; // Due to inheritance, `refine` takes a `local::BInc` (virtual methods cannot be templated).
    props[i].refine(*sub, has_changed2);
    has_changed.tell(has_changed2);
  }

  // Functions forwarded to the sub-domain `A`.

  /** `true` if the underlying abstract element is top, `false` otherwise. */
  CUDA local::BInc is_top() const {
    return sub->is_top();
  }

  /** `true` if the underlying abstract element is bot and there is no refinement function, `false` otherwise. */
  CUDA local::BDec is_bot() const {
    return sub->is_bot() && props.size() == 0;
  }

  CUDA const universe_type& operator[](int x) const {
    return (*sub)[x];
  }

  CUDA const universe_type& project(AVar x) const {
    return sub->project(x);
  }

  CUDA size_t vars() const {
    return sub->vars();
  }

  template <class Alloc2 = allocator_type>
  CUDA snapshot_type<Alloc2> snapshot(const Alloc2& alloc = Alloc2()) const {
    return snapshot_type<Alloc2>(props.size(), sub->snapshot(alloc));
  }

  template <class Alloc2>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    int n = props.size();
    for(int i = snap.num_props; i < n; ++i) {
      props.pop_back();
    }
    sub->restore(snap.sub_snap);
  }

  /** An abstract element is extractable when it is not equal to top, if all propagators are entailed and if the underlying abstract element is extractable. */
  template <class ExtractionStrategy = NonAtomicExtraction>
  CUDA bool is_extractable(const ExtractionStrategy& strategy = ExtractionStrategy()) const {
    if(is_top()) {
      return false;
    }
    for(int i = 0; i < props.size(); ++i) {
      if(!props[i].ask(*sub)) {
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
    for(int i = 0; i < props.size(); ++i) {
      seq.push_back(props[i].deinterpret(env.get_allocator(), aty()));
      map_avar_to_lvar(seq.back(), env);
    }
    return F::make_nary(AND, std::move(seq), aty());
  }
};

}

#endif
