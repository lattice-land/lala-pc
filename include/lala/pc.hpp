// Copyright 2021 Pierre Talbot

#ifndef IPC_HPP
#define IPC_HPP

#include "battery/vector.hpp"
#include "battery/unique_ptr.hpp"
#include "battery/shared_ptr.hpp"
#include "battery/allocator.hpp"

#include "lala/logic/logic.hpp"
#include "lala/universes/primitive_upset.hpp"
#include "lala/abstract_deps.hpp"

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
    using sub_snap_type = typename A::snapshot_type<Alloc2>;
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
  template<class Alloc, class F> using fresult = IResult<pc::Formula<A, Alloc>, F>;
  template<class Alloc, class F> using tresult = IResult<pc::Term<A, Alloc>, F>;

  AType atype;
  sub_ptr sub;
  battery::vector<formula_type, allocator_type> props;

public:
  template <class Alloc2, class SubTellType>
  struct interpreted_type {
    battery::vector<SubTellType, Alloc2> sub_tells;
    battery::vector<pc::Formula<A, Alloc2>, Alloc2> props;

    interpreted_type(interpreted_type&&) = default;
    interpreted_type& operator=(interpreted_type&&) = default;
    interpreted_type(const interpreted_type&) = default;

    CUDA interpreted_type(const SubTellType& sub_tell, const Alloc2& alloc = Alloc2())
      : sub_tells(alloc), props(alloc)
    {
      sub_tells.reserve(1);
      sub_tells.push_back(sub_tell);
    }

    CUDA interpreted_type(const Alloc2& alloc = Alloc2())
      : sub_tells(alloc), props(alloc) {}

    template <class InterpretedType>
    CUDA interpreted_type(const InterpretedType& other, const Alloc2& alloc = Alloc2())
      : sub_tells(other.sub_tells, alloc)
      , props(other.props, alloc)
    {}

    template <class Alloc3, class SubType2>
    friend struct interpreted_type;
  };

  template <class Alloc2>
  using tell_type = interpreted_type<Alloc2, typename sub_type::template tell_type<Alloc2>>;

  template <class Alloc2>
  using ask_type = interpreted_type<Alloc2, typename sub_type::template ask_type<Alloc2>>;

  template<class F, class Env>
  using iresult_tell = IResult<tell_type<typename Env::allocator_type>, F>;

  template<class F, class Env>
  using iresult_ask = IResult<ask_type<typename Env::allocator_type>, F>;

public:
  CUDA PC(AType atype, sub_ptr sub, const allocator_type& alloc = allocator_type())
   : atype(atype), sub(std::move(sub)), props(alloc)  {}

  CUDA PC(PC&& other)
    : atype(other.atype)
    , props(std::move(other.props))
    , sub(std::move(other.sub))
  {}

  template<class A2, class Alloc2, class... Allocators>
  CUDA PC(const PC<A2, Alloc2>& other, AbstractDeps<Allocators...>& deps)
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
  CUDA static tresult<Alloc, F> interpret_unary(const F& f, term_ptr<Alloc>&& a) {
    using T = pc::Term<A, Alloc>;
    Alloc alloc = a.get_allocator();
    switch(f.sig()) {
      case NEG: return T::make_neg(std::move(a));
      default: return tresult<Alloc, F>(IError<F>(true, name, "Unsupported unary symbol.", f));
    }
  }

  template<class Alloc, class F>
  CUDA static tresult<Alloc, F> interpret_binary(const F& f, term_ptr<Alloc>&& x, term_ptr<Alloc>&& y)
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
      default: return tresult<Alloc, F>(IError<F>(true, name, "Unsupported binary symbol.", f));
    }
  }

  template<class Alloc, class F>
  CUDA static tresult<Alloc, F> interpret_nary(const F& f, battery::vector<pc::Term<A, Alloc>, Alloc>&& subterms)
  {
    using T = pc::Term<A, Alloc>;
    switch(f.sig()) {
      case ADD: return T::make_naryadd(std::move(subterms));
      case MUL: return T::make_narymul(std::move(subterms));
      default: return tresult<Alloc, F>(IError<F>(true, name, "Unsupported nary symbol.", f));
    }
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA tresult<Alloc, F> interpret_sequence(const F& f, Env& env)
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
        t.value() = T::make_formula(
          battery::allocate_unique<pc::Formula<A, Alloc>>(alloc, std::move(p.value())));
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

  template <class F, class Env>
  CUDA IResult<AVar, F> interpret_var(const F& f, Env& env) {
    if(f.is(F::V)) {
      assert(!f.v().is_untyped());
      return IResult<AVar, F>(f.v());
    }
    else {
      auto x = var_in(f, env);
      if(!x.has_value()) {
        return IResult<AVar, F>(IError<F>(true, name, "The variable is not declared in the environment, and thus cannot be accessed by the propagator.", f));
      }
      auto avar = x->avar_of(sub->aty());
      if(avar.has_value()) {
        return IResult<AVar, F>(std::move(*avar));
      }
      else {
        return IResult<AVar, F>(IError<F>(true, name, "The variable is not declared in the sub abstract domain of PC, and thus cannot be accessed by the propagator.", f));
      }
    }
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA tresult<Alloc, F> interpret_term(const F& f, Env& env) {
    using T = pc::Term<A, Alloc>;
    Alloc alloc = env.get_allocator();
    if(f.is_variable()) {
      auto avar = interpret_var(f, env);
      if(avar.has_value()) {
        return T::make_var(avar.value());
      }
      else {
        return std::move(avar).template map_error<pc::Term<A, Alloc>>();
      }
    }
    else if(f.is_constant()) {
      using F2 = TFormula<typename Env::allocator_type>;
      auto constant = F2::make_binary(F2::make_avar(AVar()), EQ, f, UNTYPED, env.get_allocator());
      auto k = is_tell ? universe_type::interpret_tell(constant, env) : universe_type::interpret_ask(constant, env);
      if(k.has_value()) {
        return std::move(tresult<Alloc, F>(T::make_constant(std::move(k.value())))
          .join_warnings(std::move(k)));
      }
      else {
        return std::move(tresult<Alloc, F>(IError<F>(true, name, "Constant in a term could not be interpreted in the underlying abstract universe.", f))
          .join_errors(std::move(k)));
      }
    }
    else if(f.is(F::Seq)) {
      return interpret_sequence<is_tell, F>(f, env);
    }
    else {
      return tresult<Alloc, F>(IError<F>(true, name, "The shape of the formula is not supported in PC, and could not be interpreted as a term.", f));
    }
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_negation(const F& f, Env& env, bool neg_context) {
    auto nf = negate(f);
    if(nf.has_value()) {
      return interpret_formula<is_tell>(*nf, env, neg_context);
    }
    else {
      return fresult<Alloc, F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not compute its negation.", f));
    }
  }

  template<bool is_tell, class Create, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_binary_logical_connector(const F& f, const F& g, Env& env, bool neg_context, Create&& create) {
    using PF = pc::Formula<A, Alloc>;
    Alloc alloc = env.get_allocator();
    auto l = interpret_formula<is_tell>(f, env, neg_context);
    if(l.has_value()) {
      auto k = interpret_formula<is_tell>(g, env, neg_context);
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

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_conjunction(const F& f, const F& g, Env& env, bool neg_context) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<is_tell>(f, g, env, neg_context,
      [](auto&& l, auto&& k) { return PF::make_conj(std::move(l), std::move(k)); });
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_disjunction(const F& f, const F& g, Env& env) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<is_tell>(f, g, env, true,
      [](auto&& l, auto&& k) { return PF::make_disj(std::move(l), std::move(k)); });
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_biconditional(const F& f, const F& g, Env& env) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<is_tell>(f, g, env, true,
      [](auto&& l, auto&& k) { return PF::make_bicond(std::move(l), std::move(k)); });
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_implication(const F& f, const F& g, Env& env) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<is_tell>(f, g, env, true,
      [](auto&& l, auto&& k) { return PF::make_imply(std::move(l), std::move(k)); });
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_xor(const F& f, const F& g, Env& env) {
    using PF = pc::Formula<A, Alloc>;
    return interpret_binary_logical_connector<is_tell>(f, g, env, true,
      [](auto&& l, auto&& k) { return PF::make_xor(std::move(l), std::move(k)); });
  }

  template <bool neg, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_literal(const F& f, Env& env) {
    using PF = pc::Formula<A, Alloc>;
    auto avar = interpret_var(f, env);
    if(avar.has_value()) {
      if constexpr(neg) {
        return PF::make_nvarlit(std::move(avar.value()));
      }
      else {
        return PF::make_pvarlit(std::move(avar.value()));
      }
    }
    else {
      return std::move(avar).template map_error<pc::Formula<A, Alloc>>();
    }
  }

  /** expr != k is transformed into expr < k \/ expr > k.
   * `k` needs to be an integer. */
  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_neq_decomposition(const F& f, Env& env, bool neg_context) {
    if(f.sig() == NEQ && f.seq(1).is(F::Z)) {
      using F2 = TFormula<typename Env::allocator_type>;
      typename Env::allocator_type alloc = env.get_allocator();
      return interpret_formula<is_tell>(
        F2::make_binary(
          F2::make_binary(f.seq(0), LT, f.seq(1), f.type(), alloc),
          OR,
          F2::make_binary(f.seq(0), GT, f.seq(1), f.type(), alloc),
          f.type(),
          alloc),
        env,
        neg_context
      );
    }
    else {
      return fresult<Alloc, F>(IError<F>(true, name, "Unsupported predicate in this abstract domain.", f));
    }
  }

  template <bool is_tell, class F, class Env, class Alloc = typename Env::allocator_type>
  CUDA fresult<Alloc, F> interpret_formula(const F& f, Env& env, bool neg_context = false) {
    assert(f.type() == aty() || f.is_untyped() || f.is_variable());
    Alloc alloc = env.get_allocator();
    using PF = pc::Formula<A, Alloc>;
    if(f.is_false()) {
      return PF::make_false();
    }
    else if(f.is_true()) {
      return PF::make_true();
    }
    else if(f.is_binary()) {
      Sig sig = f.sig();
      switch(sig) {
        case AND: return interpret_conjunction<is_tell>(f.seq(0), f.seq(1), env, neg_context);
        case OR:  return interpret_disjunction<is_tell>(f.seq(0), f.seq(1), env);
        case EQUIV: return interpret_biconditional<is_tell>(f.seq(0), f.seq(1), env);
        case IMPLY: return interpret_implication<is_tell>(f.seq(0), f.seq(1), env);
        case XOR: return interpret_xor<is_tell>(f.seq(0), f.seq(1), env);
        case EQ: // Whenever an operand of `=` is a formula with logical connectors, we interpret `=` as `<=>`.
          if(f.seq(0).is_logical() || f.seq(1).is_logical()) {
            return interpret_biconditional<is_tell>(f.seq(0), f.seq(1), env);
          }
        // Form of the constraint `T <op> u` with `x <op> u` interpreted in the underlying universe.
        default:
          auto fn = move_constants_on_rhs(f);
          using F2 = TFormula<typename Env::allocator_type>;
          auto fu = F2::make_binary(F2::make_avar(AVar()), fn.sig(), fn.seq(1), fn.type(), alloc);
          auto u = is_tell ? universe_type::interpret_tell(fu, env) : universe_type::interpret_ask(fu, env);
          if(!u.has_value() && fn.sig() == NEQ) {
            return interpret_neq_decomposition<is_tell>(fn, env, neg_context);
          }
          else if(u.has_value()) {
            auto term = interpret_term<is_tell>(fn.seq(0), env);
            if(term.has_value()) {
              // In a context where the formula propagator can be asked for its negation, we must interpret the negation of the formula as well.
              if(neg_context) {
                auto nf_ = negate(fn);
                if(nf_.has_value()) {
                  auto nf = interpret_formula<is_tell>(*nf_, env);
                  if(nf.has_value()) {
                    auto data = PF::make_nlop(std::move(term.value()), std::move(u.value()),
                      battery::allocate_unique<PF>(alloc, std::move(nf.value())));
                    return std::move(fresult<Alloc, F>(std::move(data))
                      .join_warnings(std::move(nf))
                      .join_warnings(std::move(u))
                      .join_warnings(std::move(term)));
                  }
                  else {
                    return std::move(fresult<Alloc, F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not interpret its negation.", f))
                      .join_errors(std::move(nf))
                      .join_warnings(std::move(u))
                      .join_warnings(std::move(term)));
                  }
                }
                else {
                  return std::move(fresult<Alloc, F>(IError<F>(true, name, "We must query this formula for disentailement, but we could not compute its negation.", f))
                    .join_warnings(std::move(u))
                    .join_warnings(std::move(term)));
                }
              }
              else {
                auto data = PF::make_plop(std::move(term.value()), std::move(u.value()));
                return std::move(fresult<Alloc, F>(std::move(data))
                  .join_warnings(std::move(u))
                  .join_warnings(std::move(term)));
              }
            }
            else {
              return std::move(fresult<Alloc, F>(IError<F>(true, name, "We cannot interpret the term on the LHS of the formula in PC.", f))
                .join_warnings(std::move(u))
                .join_errors(std::move(term)));
            }
          }
          else {
            return std::move(fresult<Alloc, F>(IError<F>(true, name, "We cannot interpret the constant on the RHS of the formula in the underlying abstract universe.", f))
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
      return interpret_negation<is_tell>(f, env, neg_context);
    }
    // Singleton formula
    else if(f.is(F::Seq) && f.seq().size() == 1) {
      return interpret_formula<is_tell>(f.seq(0), env, neg_context);
    }
    else {
      return fresult<Alloc, F>(IError<F>(true, name, "The shape of this formula is not supported.", f));
    }
  }

  template <bool is_tell, class R, class F, class Env>
  CUDA void interpret_formula2(const F& f, Env& env, R& res) {
    auto interpreted = interpret_formula<is_tell>(f, env);
    if(interpreted.has_value()) {
      res.value().props.push_back(std::move(interpreted.value()));
      res.join_warnings(std::move(interpreted));
    }
    else {
      res = std::move(R(IError<F>(true, name,
          "A formula typed in PC (or untyped) could not be interpreted.", f))
        .join_warnings(std::move(res))
        .join_errors(std::move(interpreted)));
    }
  }

  template <class R, bool is_tell, class F, class Env>
  CUDA R interpret_in(const F& f, Env& env) {
    using val_t = typename R::value_type;
    // If the formula is untyped, we first try to interpret it in the sub-domain.
    IResult<int, F> sub_err{0};
    if(f.is_untyped() || f.type() != aty()) {
      auto r = is_tell ? sub->interpret_tell_in(f, env) : sub->interpret_ask_in(f, env);
      if(r.has_value()) {
        return std::move(r).map(val_t(std::move(r.value()), env.get_allocator()));
      }
      if(f.type() != UNTYPED) {
        return std::move(R(IError<F>(true, name,
            "The formula could not be interpreted in the sub-domain, and it has an abstract type different from the one of the current element.", f))
          .join_errors(std::move(r)));
      }
      sub_err = IResult<int, F>(std::move(r.error()));
    }
    assert(f.is_untyped() || f.type() == aty());
    // Conjunction
    if(f.is(F::Seq) && f.sig() == AND) {
      auto split_formula = extract_ty(f, aty());
      const auto& ipc_formulas = battery::get<0>(split_formula);
      const auto& other_formulas = battery::get<1>(split_formula);
      R res(val_t(env.get_allocator()));
      // We need to interpret the formulas in the sub-domain first because it might handle existential quantifiers needed by formulas interpreted in this domain.
      for(int i = 0; i < other_formulas.seq().size(); ++i) {
        auto sub_tell = is_tell ? sub->interpret_tell_in(other_formulas.seq(i), env) : sub->interpret_ask_in(other_formulas.seq(i), env);
        if(sub_tell.has_value()) {
          res.value().sub_tells.push_back(std::move(sub_tell.value()));
          res.join_warnings(std::move(sub_tell));
        }
        else if(other_formulas.seq(i).type() == UNTYPED) {
          interpret_formula2<is_tell>(other_formulas.seq(i), env, res);
          if(!res.has_value()) {
            res.join_errors(std::move(sub_tell));
            return std::move(res);
          }
        }
        else {
          return std::move(R(IError<F>(true, name,
              "The formula could not be interpreted in the sub-domain and it has an abstract type different from the one of the current element.", f))
            .join_errors(std::move(sub_tell)));
        }
      }
      for(int i = 0; i < ipc_formulas.seq().size(); ++i) {
        interpret_formula2<is_tell>(ipc_formulas.seq(i), env, res);
        if(!res.has_value()) {
          return std::move(std::move(res).join_errors(std::move(sub_err)));
        }
      }
      return std::move(res);
    }
    else {
      R res(val_t(env.get_allocator()));
      interpret_formula2<is_tell>(f, env, res);
      if(!res.has_value()) {
        return std::move(std::move(res).join_errors(std::move(sub_err)));
      }
      return std::move(res);
    }
  }

public:
  /** PC expects a conjunction of the form \f$ c_1 \land \ldots \land c_n \f$ where sub-formulas \f$ c_i \f$ can either be interpreted in the sub-domain `A` or in the current domain.
    Moreover, we only treat exact conjunction (no under- or over-approximation of the conjunction).
    For now, \f$ T \neq k \f$ is not supported where \f$ T \f$ is an arithmetic term, containing function symbols supported in `terms.hpp`. */
  template <class F, class Env>
  CUDA iresult_tell<F, Env> interpret_tell_in(const F& f, Env& env) {
    return interpret_in<iresult_tell<F, Env>, true>(f, env);
  }

  /** Create an abstract domain and interpret the formulas `f` in this abstract domain.
   * The sub abstract domain is supposed to be able to represent variables, and its constructor is assumed to take a size, like for `VStore`. */
  template <class F, class Env>
  CUDA static IResult<this_type, F> interpret_tell(const F& f, Env& env, allocator_type alloc = allocator_type()) {
    this_type ipc(env.extends_abstract_dom(),
      battery::allocate_shared<sub_type>(alloc, env.extends_abstract_dom(), num_quantified_untyped_vars(f)),
      alloc);
    auto r = ipc.interpret_tell_in(f, env);
    if(r.has_value()) {
      ipc.tell(r.value());
      return std::move(IResult<this_type, F>(std::move(ipc)).join_warnings(std::move(r)));
    }
    else {
      return std::move(r).template map_error<this_type>();
    }
  }

  template <class F, class Env>
  CUDA iresult_ask<F, Env> interpret_ask_in(const F& f, const Env& env) const {
    return interpret_in<iresult_ask<F, Env>, false>(f, env);
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

  template <class Alloc2 = allocator_type>
  CUDA void restore(const snapshot_type<Alloc2>& snap) {
    int n = props.size();
    for(int i = snap.num_props; i < n; ++i) {
      props.pop_back();
    }
    sub->restore(snap.sub_snap);
  }

  /** Extract an under-approximation of `this` in `ua` when all propagators are entailed.
   * In all other cases, returns `false`.
   * For efficiency reason, if `B` is a propagator completion, the propagators are not copied in `ua`.
   *   (It is OK, since they are entailed, so don't bring information anymore.) */
  template <class B>
  CUDA bool extract(B& ua) const {
    if(is_top()) {
      return false;
    }
    for(int i = 0; i < props.size(); ++i) {
      if(!props[i].ask(*sub)) {
        return false;
      }
    }
    if constexpr(impl::is_pc_like<B>::value) {
      return sub->extract(*ua.sub);
    }
    else {
      return sub->extract(ua);
    }
  }
};

}

#endif
