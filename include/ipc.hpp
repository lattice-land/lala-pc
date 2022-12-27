// Copyright 2021 Pierre Talbot

#ifndef IPC_HPP
#define IPC_HPP

#include "logic/logic.hpp"
#include "universes/upset_universe.hpp"
#include "copy_dag_helper.hpp"

#include "terms.hpp"
#include "formula.hpp"

#include "vector.hpp"
#include "unique_ptr.hpp"
#include "shared_ptr.hpp"

namespace lala {

/** IPC is an abstract transformer built on top of an abstract domain `A`.
    It is expected that `A` has a projection function `u = project(x)`.
    We also expect a `tell(x, u, has_changed)` function to join the abstract universe `u` in the domain of the variable `x`.
    An example of abstract domain satisfying these requirements is `VStore<Interval<ZInc>>`. */
template <class A, class Alloc = typename A::allocator_type>
class IPC {
public:
  using sub_type = A;
  using universe_type = typename A::universe_type;
  using allocator_type = Alloc;
  using this_type = IPC<A, allocator_type>;

  template <class Alloc2>
  using snapshot_type = battery::tuple<size_t, typename A::snapshot_type<Alloc2>>;

  using sub_ptr = battery::shared_ptr<A, allocator_type>;

  constexpr static const char* name = "IPC";

private:
  using formula_type = battery::shared_ptr<Formula<A>, allocator_type>;
  using term_type = battery::shared_ptr<Term<A>, allocator_type>;

  AType atype;
  sub_ptr sub;
  battery::vector<formula_type, allocator_type> props;

public:
  template <class Alloc2>
  struct tell_type {
    using sub_tell_type = typename sub_type::tell_type<Alloc2>;
    battery::vector<sub_tell_type, Alloc2> sub_tells;
    battery::vector<formula_type, Alloc2> props;
    CUDA tell_type(tell_type&&) = default;
    CUDA tell_type& operator=(tell_type&&) = default;
    CUDA tell_type(const tell_type&) = default;
    CUDA tell_type(sub_tell_type&& sub_tell, const Alloc2& alloc): sub_tells(alloc), props(alloc) {
      sub_tells.push_back(std::move(sub_tell));
    }
    CUDA tell_type(size_t n, const Alloc2& alloc): sub_tells(alloc), props(alloc) {
      props.reserve(n);
    }
  };

  template<class F, class Env>
  using iresult = IResult<tell_type<typename Env::allocator_type>, F>;

public:
  CUDA IPC(AType atype, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : atype(atype), sub(std::move(sub)), props(alloc)  {}

  CUDA IPC(IPC&& other): atype(other.atype), props(std::move(other.props)) {
    sub.swap(other.sub);
  }

  /** The propagators are shared among IPC, it currently works because propagators are stateless.
   * This could be changed later on by adding a method clone in `Term`. */
  template<class Alloc2, class Alloc3>
  CUDA IPC(const IPC<A, Alloc2>& other, AbstractDeps<allocator_type, Alloc3>& deps)
   : atype(other.atype), sub(deps.clone(other.sub)), props(other.props, deps.get_allocator())
  {}

  CUDA allocator_type get_allocator() const {
    return props.get_allocator();
  }

  CUDA AType aty() const {
    return atype;
  }

  CUDA static this_type bot(AType atype = UNTYPED, const allocator_type& alloc = allocator_type()) {
    return IPC(atype, battery::allocate_shared<sub_type>(alloc, std::move(sub_type::bot(UNTYPED, alloc))), alloc);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType atype = UNTYPED, const allocator_type& alloc = allocator_type()) {
    return IPC(atype, battery::allocate_shared<sub_type>(alloc, std::move(sub_type::top(UNTYPED, alloc))), alloc);
  }

private:

  template<class F>
  using tresult = IResult<term_type, F>;

  template<class Arg>
  CUDA term_type make_ptr(Arg&& arg) {
    return battery::allocate_shared<DynTerm<Arg>>(get_allocator(), std::move(arg));
  }

  template<class F>
  CUDA tresult<F> interpret_unary(const F& f, term_type&& a) {
    switch(f.sig()) {
      case NEG: return make_ptr(Neg<term_type>(std::move(a)));
      default: return tresult<F>(IError<F>(true, name, "Unsupported unary symbol.", f));
    }
  }

  template<class F>
  CUDA tresult<F> interpret_binary(const F& f, term_type&& x, term_type&& y)
  {
    switch(f.sig()) {
      case ADD: return make_ptr(Add<term_type, term_type>(std::move(x), std::move(y)));
      case SUB: return make_ptr(Sub<term_type, term_type>(std::move(x), std::move(y)));
      case MUL: return make_ptr(Mul<term_type, term_type>(std::move(x), std::move(y)));
      case DIV: return make_ptr(Div<term_type, term_type>(std::move(x), std::move(y)));
      default: return tresult<F>(IError<F>(true, name, "Unsupported binary symbol.", f));
    }
  }

  template<class F>
  CUDA tresult<F> interpret_nary(const F& f, battery::vector<term_type, allocator_type>&& subterms)
  {
    switch(f.sig()) {
      case ADD: return make_ptr(NaryAdd<term_type, allocator_type>(std::move(subterms)));
      case MUL: return make_ptr(NaryMul<term_type, allocator_type>(std::move(subterms)));
      default: return tresult<F>(IError<F>(true, name, "Unsupported nary symbol.", f));
    }
  }

  template <class F, class Env>
  CUDA tresult<F> interpret_sequence(const F& f, Env& env)
  {
    battery::vector<term_type, allocator_type> subterms;
    subterms.reserve(f.seq().size());
    for(int i = 0; i < f.seq().size(); ++i) {
      auto t = interpret_term(f.seq(i), env);
      if(!t.has_value()) {
        auto p = interpret_formula(f.seq(i), env, true);
        if(!p.has_value()) {
          return std::move(t.join_errors(std::move(p)));
        }
        t.value() = p.value();
      }
      subterms.push_back(std::move(t.value()));
    }
    if(subterms.size() == 1) {
      return interpret_unary(f, std::move(subterms[0]));
    }
    else if(subterms.size() == 2) {
      return interpret_binary(f, std::move(subterms[0]), std::move(subterms[1]));
    }
    else {
      return interpret_nary(f, std::move(subterms));
    }
  }

  template <class F, class Env>
  CUDA IResult<AVar, F> interpret_var(const F& f, Env& env) {
    auto x = var_in(f, env);
    assert(x.has_value());
    auto avar = x->avar_of(sub->aty());
    if(avar.has_value()) {
      return IResult<AVar, F>(std::move(*avar));
    }
    else {
      return IResult<AVar, F>(IError<F>(true, name, "The variable is not declared in the sub abstract domain of IPC, and thus cannot be accessed by the propagator.", f));
    }
  }

  template <class F, class Env>
  CUDA tresult<F> interpret_term(const F& f, Env& env) {
    if(f.is_variable()) {
      auto avar = interpret_var(f, env);
      if(avar.has_value()) {
        return make_ptr(Variable<A>(std::move(avar.value())));
      }
      else {
        return std::move(avar).template map_error<term_type>();
      }
    }
    else if(f.is_constant()) {
      auto k = universe_type::interpret(F::make_binary(F::make_avar(AVar()), EQ, f), env);
      if(k.has_value()) {
        return std::move(tresult<F>(make_ptr(Constant<A>(std::move(k.value()))))
          .join_warnings(std::move(k)));
      }
      else {
        return std::move(tresult<F>(IError<F>(true, name, "Constant in a term could not be interpreted in the underlying abstract universe.", f))
          .join_errors(std::move(k)));
      }
    }
    else if(f.is(F::Seq)) {
      return interpret_sequence<F>(f, env);
    }
    else {
      return tresult<F>(IError<F>(true, name, "The shape of the formula is not supported in IPC, and could not be interpreted as a term.", f));
    }
  }

  template<class F>
  using fresult = IResult<formula_type, F>;

  template<class Arg>
  CUDA formula_type make_fptr(Arg&& arg) {
    return battery::allocate_shared<DynFormula<Arg>>(props.get_allocator(), std::move(arg));
  }

  template <class F, class Env>
  CUDA fresult<F> interpret_negation(const F& f, Env& env, bool neg_context) {
    auto nf = negate(f);
    if(nf.has_value()) {
      return interpret_formula(*nf, env, neg_context);
    }
    else {
      return fresult<F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not compute its negation.", f));
    }
  }

  template<template<class> class LogicalConnector, class F, class Env>
  CUDA fresult<F> interpret_binary_logical_connector(const F& f, const F& g, Env& env, bool neg_context) {
    auto l = interpret_formula(f, env, neg_context);
    if(l.has_value()) {
      auto k = interpret_formula(g, env, neg_context);
      if(k.has_value()) {
        return std::move(fresult<F>(make_fptr(LogicalConnector<formula_type>(std::move(l.value()), std::move(k.value()))))
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

  template <class F, class Env>
  CUDA fresult<F> interpret_conjunction(const F& f, const F& g, Env& env, bool neg_context) {
    return interpret_binary_logical_connector<Conjunction>(f, g, env, neg_context);
  }

  template <class F, class Env>
  CUDA fresult<F> interpret_disjunction(const F& f, const F& g, Env& env) {
    return interpret_binary_logical_connector<Disjunction>(f, g, env, true);
  }

  template <class F, class Env>
  CUDA fresult<F> interpret_biconditional(const F& f, const F& g, Env& env) {
    return interpret_binary_logical_connector<Biconditional>(f, g, env, true);
  }

  template <bool neg, class F, class Env>
  CUDA fresult<F> interpret_literal(const F& f, Env& env) {
    auto avar = interpret_var(f, env);
    if(avar.has_value()) {
      return make_fptr(VariableLiteral<A, neg>(std::move(avar.value())));
    }
    else {
      return std::move(avar).template map_error<formula_type>();
    }
  }

  template <class F, class Env>
  CUDA fresult<F> interpret_formula(const F& f, Env& env, bool neg_context = false) {
    assert(f.type() == aty() || f.type() == UNTYPED);
    if(f.is_binary()) {
      Sig sig = f.sig();
      switch(sig) {
        case AND: return interpret_conjunction(f.seq(0), f.seq(1), env, neg_context);
        case OR:  return interpret_disjunction(f.seq(0), f.seq(1), env);
        case EQUIV:  return interpret_biconditional(f.seq(0), f.seq(1), env);
        // Form of the constraint `T <op> u` with `x <op> u` interpreted in the underlying universe.
        default:
          auto fn = move_constants_on_rhs(f);
          auto u = universe_type::interpret(F::make_binary(F::make_avar(AVar()), fn.sig(), fn.seq(1)), env);
          if(u.has_value()) {
            auto term = interpret_term(fn.seq(0), env);
            if(term.has_value()) {
              // In a context where the formula propagator can be asked for its negation, we must interpret the negation of the formula as well.
              if(neg_context) {
                auto nf_ = negate(fn);
                if(nf_.has_value()) {
                  auto nf = interpret_formula(*nf_, env);
                  if(nf.has_value()) {
                    auto data = make_fptr(LatticeOrderPredicate<term_type, formula_type>(std::move(term.value()), std::move(u.value()), std::move(nf.value())));
                    return std::move(fresult<F>(std::move(data))
                      .join_warnings(std::move(nf))
                      .join_warnings(std::move(u))
                      .join_warnings(std::move(term)));
                  }
                  else {
                    return std::move(fresult<F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not interpret its negation.", f))
                      .join_errors(std::move(nf))
                      .join_warnings(std::move(u))
                      .join_warnings(std::move(term)));
                  }
                }
                else {
                  return std::move(fresult<F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not compute its negation.", f))
                    .join_warnings(std::move(u))
                    .join_warnings(std::move(term)));
                }
              }
              else {
                auto data = make_fptr(LatticeOrderPredicate<term_type>(std::move(term.value()), std::move(u.value())));
                return std::move(fresult<F>(std::move(data))
                  .join_warnings(std::move(u))
                  .join_warnings(std::move(term)));
              }
            }
            else {
              return std::move(fresult<F>(IError<F>(true, name, "We cannot interpret the term on the LHS of the formula in IPC.", f))
                .join_warnings(std::move(u))
                .join_errors(std::move(term)));
            }
          }
          else {
            return std::move(fresult<F>(IError<F>(true, name, "We cannot interpret the constant on the RHS of the formula in the underlying abstract universe.", f))
              .join_errors(std::move(u)));
          }
      }
    }
    // Negative literal
    else if(f.is(F::Seq) && f.seq().size() == 1 && f.sig() == NOT &&
      f.seq(0).is_variable())
    {
      return interpret_literal<true>(f.seq(0), env);
    }
    // Positive literal
    else if(f.is_variable()) {
      return interpret_literal<false>(f, env);
    }
    // Logical negation
    else if(f.is(F::Seq) && f.seq().size() == 1 && f.sig() == NOT) {
      return interpret_negation(f, env, neg_context);
    }
    // Singleton formula
    else if(f.is(F::Seq) && f.seq().size() == 1) {
      return interpret_formula(f.seq(0), env, neg_context);
    }
    else {
      return fresult<F>(IError<F>(true, name, "The shape of this formula is not supported.", f));
    }
  }

  template <class F, class Env>
  CUDA void interpret_formula2(const F& f, Env& env, iresult<F, Env>& res) {
    auto ipc_tell = interpret_formula(f, env);
    if(ipc_tell.has_value()) {
      res.value().props.push_back(std::move(ipc_tell.value()));
      res.join_warnings(std::move(ipc_tell));
    }
    else {
      res = std::move(iresult<F, Env>(IError<F>(true, name,
          "A formula typed in IPC (or untyped) could not be interpreted.", f))
        .join_warnings(std::move(res))
        .join_errors(std::move(ipc_tell)));
    }
  }

public:
  /** IPC expects a conjunction of the form \f$ c_1 \land \ldots \land c_n \f$ where sub-formulas \f$ c_i \f$ can either be interpreted in the sub-domain `A` or in the current domain.
    Moreover, we only treat exact conjunction (no under- or over-approximation of the conjunction).
    For now, \f$ T \neq k \f$ is not supported where \f$ T \f$ is an arithmetic term, containing function symbols supported in `terms.hpp`. */
  template <class F, class Env>
  CUDA iresult<F, Env> interpret_in(const F& f, Env& env) {
    using tell_t = tell_type<typename Env::allocator_type>;
    // If the formula is untyped, we first try to interpret it in the sub-domain.
    if(f.type() == UNTYPED || f.type() != aty()) {
      auto r = sub->interpret_in(f, env);
      if(r.has_value()) {
        return std::move(r).map(tell_t(std::move(r.value()), env.get_allocator()));
      }
      if(f.type() != UNTYPED) {
        return std::move(iresult<F, Env>(IError<F>(true, name,
            "The formula could not be interpreted in the sub-domain, and it has an abstract type different from the one of the current element.", f))
          .join_errors(std::move(r)));
      }
    }
    assert(f.type() == UNTYPED || f.type() == aty());
    // Conjunction
    if(f.is(F::Seq) && f.sig() == AND) {
      auto split_formula = extract_ty(f, aty());
      const auto& ipc_formulas = battery::get<0>(split_formula).seq();
      const auto& other_formulas = battery::get<1>(split_formula);
      iresult<F, Env> res(tell_t(ipc_formulas.size(), env.get_allocator()));
      // We need to interpret the formulas in the sub-domain first because it might handle existential quantifiers needed by formulas interpreted in this domain.
      for(int i = 0; i < other_formulas.seq().size(); ++i) {
        typename sub_type::iresult<F, Env> sub_tell = sub->interpret_in(other_formulas.seq(i), env);
        if(sub_tell.has_value()) {
          res.value().sub_tells.push_back(std::move(sub_tell.value()));
          res.join_warnings(std::move(sub_tell));
        }
        else if(other_formulas.seq(i).type() == UNTYPED) {
          interpret_formula2(other_formulas.seq(i), env, res);
          if(!res.has_value()) {
            res.join_errors(std::move(sub_tell));
            return std::move(res);
          }
        }
        else {
          return std::move(iresult<F, Env>(IError<F>(true, name,
              "The formula could not be interpreted in the sub-domain and it has an abstract type different from the one of the current element.", f))
            .join_errors(std::move(sub_tell)));
        }
      }
      for(int i = 0; i < ipc_formulas.size(); ++i) {
        interpret_formula2(other_formulas.seq(i), env, res);
        if(!res.has_value()) {
          return std::move(res);
        }
      }
      return std::move(res);
    }
    else {
      iresult<F, Env> res(tell_t(1, env.get_allocator()));
      interpret_formula2(f, env, res);
      return std::move(res);
    }
  }

  /** Create an abstract domain and interpret the formulas `f` in this abstract domain.
   * The sub abstract domain is supposed to be able to represent variables, and its constructor is assumed to take a size, like for `VStore`. */
  template <class F, class Env>
  CUDA static IResult<this_type, F> interpret(const F& f, Env& env, allocator_type alloc = allocator_type()) {
    this_type ipc(env.extends_abstract_dom(),
      battery::allocate_shared<sub_type>(alloc, env.extends_abstract_dom(), num_quantified_untyped_vars(f)),
      alloc);
    iresult<F, Env> r = ipc.interpret_in(f, env);
    if(r.has_value()) {
      ipc.tell(std::move(r.value()));
      return std::move(IResult<this_type, F>(std::move(ipc)).join_errors(std::move(r)));
    }
    else {
      return std::move(r).template map_error<this_type>();
    }
  }


  /** Note that we cannot add propagators in parallel (but modifying the underlying domain is ok).
      This is a current limitation that we should fix later on.
      Notes for later:
        * To implement "telling of propagators", we would need to check if a propagator has already been added or not (for idempotency).
        * 1. Walk through the existing propagators to check which ones are already in.
        * 2. If a propagator has the same shape but different constant `U`, join them in place.  */
  template <class Alloc2, class Mem>
  CUDA this_type& tell(tell_type<Alloc2>&& t, BInc<Mem>& has_changed) {
    for(int i = 0; i < t.sub_tells.size(); ++i) {
      sub->tell(t.sub_tells[i], has_changed);
    }
    if(t.props.size() > 0) {
      has_changed.tell_top();
    }
    size_t n = props.size();
    props.reserve(n + t.props.size());
    for(int i = 0; i < t.props.size(); ++i) {
      props.push_back(std::move(t.props[i]));
      props[n + i]->preprocess(*sub, has_changed);
    }
    return *this;
  }

  template <class Alloc2>
  CUDA this_type& tell(tell_type<Alloc2>&& t) {
    local::BInc has_changed;
    return tell(std::move(t), has_changed);
  }

  template <class Alloc2>
  CUDA local::BInc ask(const tell_type<Alloc2>& t) const {
    for(int i = 0; i < t.props.size(); ++i) {
      if(!t.props[i]->ask(*sub)) {
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
    props[i]->refine(*sub, has_changed);
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
  CUDA snapshot_type<Alloc2> snapshot() const {
    return battery::make_tuple(props.size(), sub->template snapshot<Alloc2>());
  }

  template <class Alloc2 = allocator_type>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    int n = props.size();
    for(int i = battery::get<0>(snap); i < n; ++i) {
      props.pop_back();
    }
    sub->restore(battery::get<1>(snap));
  }

  /** Extract an under-approximation of `this` in `ua` when all propagators are entailed.
   * In all other cases, returns `false`.
   * For efficiency reason, the propagators are not copied in `ua` (it is OK, since they are entailed, so don't bring information anymore). */
  template <class A2, class Alloc2>
  CUDA bool extract(IPC<A2, Alloc2>& ua) const {
    if(is_top()) {
      return false;
    }
    for(int i = 0; i < props.size(); ++i) {
      if(!props[i]->ask(*sub)) {
        return false;
      }
    }
    return sub->extract(*ua.sub);
  }
};

}

#endif
