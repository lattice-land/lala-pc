// Copyright 2021 Pierre Talbot

#ifndef IPC_HPP
#define IPC_HPP

#include "ast.hpp"
#include "terms.hpp"
#include "formula.hpp"
#include "rewritting.hpp"
#include "z.hpp"

#include "vector.hpp"
#include "unique_ptr.hpp"
#include "shared_ptr.hpp"

namespace lala {

struct gpu_tag_t {} gpu_tag;

/** IPC is an abstract transformer built on top of an abstract domain `A`.
    It is expected that `A` has a projection function `itv = project(x)`, and the resulting abstract universe has a lower bound and upper bound projection functions `itv.lb()` and `itv.ub()`.
    We also expect a `tell(x, itv, has_changed)` function to join the interval `itv` in the domain of the variable `x`.
    An example of abstract domain satisfying these requirements is `VStore<Interval<ZInc<int>>>`. */
template <class AD, class Alloc = typename AD::Allocator>
class IPC {
public:
  using A = AD;
  using Universe = typename A::Universe;
  using Allocator = Alloc;
  using this_type = IPC<A, Allocator>;
  using Env = typename A::Env;

  using SubPtr = battery::shared_ptr<A, Allocator>;
private:
  AType uid;
  SubPtr a;
  using FormulaPtr = battery::unique_ptr<Formula<A>, Allocator>;
  using TermPtr = battery::unique_ptr<Term<A>, Allocator>;
  battery::vector<FormulaPtr, Allocator> props;

public:
  CUDA IPC(AType uid, SubPtr a,
    const Allocator& alloc = Allocator()): a(std::move(a)), props(alloc), uid(uid)  {}

  CUDA IPC(IPC&& other): props(std::move(other.props)), uid(other.uid) {
    ::battery::swap(a, other.a);
  }

  CUDA static this_type bot(AType uid = UNTYPED) {
    return IPC(uid, battery::make_shared(std::move(A::bot())));
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType uid = UNTYPED) {
    return IPC(uid, battery::make_shared(std::move(A::top())));
  }

private:
  using TermOpt = thrust::optional<TermPtr>;

  template<class Arg>
  CUDA TermPtr make_ptr(Arg&& arg) {
    return battery::allocate_unique<DynTerm<Arg>>(props.get_allocator(), std::move(arg));
  }

  CUDA TermOpt interpret_unary(Sig sig, TermPtr&& a) {
    switch(sig) {
      case NEG: return make_ptr(Neg<TermPtr>(std::move(a)));
      default:  return {};
    }
  }

  CUDA TermOpt interpret_binary(Sig sig, TermPtr&& x, TermPtr&& y)
  {
    switch(sig) {
      case ADD: return make_ptr(Add<TermPtr, TermPtr>(std::move(x), std::move(y)));
      case SUB: return make_ptr(Sub<TermPtr, TermPtr>(std::move(x), std::move(y)));
      case MUL: return make_ptr(Mul<TermPtr, TermPtr>(std::move(x), std::move(y)));
      case DIV: return make_ptr(Div<TermPtr, TermPtr>(std::move(x), std::move(y)));
      default:  return {};
    }
  }

  CUDA TermOpt interpret_nary(Sig sig, battery::vector<TermPtr, Allocator>&& subterms)
  {
    switch(sig) {
      case ADD: return make_ptr(NaryAdd<TermPtr, Allocator>(std::move(subterms)));
      case MUL: return make_ptr(NaryMul<TermPtr, Allocator>(std::move(subterms)));
      default:  return {};
    }
  }

  template <class F>
  CUDA TermOpt interpret_sequence(Sig sig, const typename F::Sequence& seq)
  {
    battery::vector<TermPtr, Allocator> subterms;
    subterms.reserve(seq.size());
    for(int i = 0; i < seq.size(); ++i) {
      auto t = interpret_term(seq[i]);
      if(!t.has_value()) {
        t = interpret_formula(seq[i], true);
        if(!t.has_value()) {
          return {};
        }
      }
      subterms.push_back(std::move(*t));
    }
    if(subterms.size() == 1) {
      return interpret_unary(sig, std::move(subterms[0]));
    }
    else if(subterms.size() == 2) {
      return interpret_binary(sig, std::move(subterms[0]), std::move(subterms[1]));
    }
    else {
      return interpret_nary(sig, std::move(subterms));
    }
  }

  template <class F>
  CUDA TermOpt interpret_term(const F& f) {
    if(f.is(F::V)) {
      // need to check that this variable is defined in the domain?
      // or should we just suppose it is?
      return make_ptr(Variable<A>(f.v()));
    }
    else if(f.is(F::LV)) {
      auto v = a->environment().to_avar(f.lv());
      if(!v.has_value()) {
        return {};
      }
      else {
        return make_ptr(Variable<A>(*v));
      }
    }
    else if(f.is(F::Z)) {
      auto k = Universe::interpret(F::make_binary(F::make_avar(0), EQ, f));
      if(!k.has_value()) {
        return {};
      }
      else {
        return make_ptr(Constant<A>(std::move(*k)));
      }
    }
    else if(f.is(F::Seq)) {
      return interpret_sequence<F>(f.sig(), f.seq());
    }
    return {};
  }

  using FormulaOpt = thrust::optional<FormulaPtr>;

  template<class Arg>
  CUDA FormulaPtr make_fptr(Arg&& arg) {
    return battery::allocate_unique<DynFormula<Arg>>(props.get_allocator(), std::move(arg));
  }

  template <class F>
  CUDA FormulaOpt interpret_negation(const F& f, bool neg_context) {
    auto nf = negate(f);
    if(nf.has_value()) {
      return interpret_formula(*nf, neg_context);
    }
    return {};
  }

  template<class F, template<class> class LogicalConnector>
  CUDA FormulaOpt interpret_binary_logical_connector(const F& f, const F& g, bool neg_context) {
    auto l = interpret_formula(f, neg_context);
    if(l.has_value()) {
      auto k = interpret_formula(g, neg_context);
      if(k.has_value()) {
        return make_fptr(LogicalConnector<FormulaPtr>(std::move(*l), std::move(*k)));
      }
    }
    return {};
  }

  template <class F>
  CUDA FormulaOpt interpret_conjunction(const F& f, const F& g, bool neg_context) {
    return interpret_binary_logical_connector<F, Conjunction>(f, g, neg_context);
  }

  template <class F>
  CUDA FormulaOpt interpret_disjunction(const F& f, const F& g) {
    return interpret_binary_logical_connector<F, Disjunction>(f, g, true);
  }

  template <class F>
  CUDA FormulaOpt interpret_biconditional(const F& f, const F& g) {
    return interpret_binary_logical_connector<F, Biconditional>(f, g, true);
  }

  template <bool neg, class F>
  CUDA FormulaOpt interpret_literal(const F& f) {
    auto x = var_in(f, a->environment());
    assert(x.has_value());
    return make_fptr(VariableLiteral<A, neg>(std::move(*x)));
  }

  template <class F>
  CUDA FormulaOpt interpret_formula(const F& f, bool neg_context = false) {
    assert(f.type() == uid || f.type() == UNTYPED);
    if(f.is(F::Seq) && f.seq().size() == 2) {
      Sig sig = f.sig();
      switch(sig) {
        case AND: return interpret_conjunction(f.seq(0), f.seq(1), neg_context);
        case OR:  return interpret_disjunction(f.seq(0), f.seq(1));
        case EQUIV:  return interpret_biconditional(f.seq(0), f.seq(1));
        // Form of the constraint `T <op> u` with `x <op> u` interpreted in the underlying universe.
        default:
          auto fn = move_constants_on_rhs(f);
          auto u = Universe::interpret(F::make_binary(F::make_avar(0), fn.sig(), fn.seq(1)));
          if(u.has_value()) {
            auto term = interpret_term(fn.seq(0));
            if(term.has_value()) {
              // In a context where the formula propagator can be asked for its negation, we must interpret the negation of the formula as well.
              if(neg_context) {
                auto nf_ = negate(fn);
                if(nf_.has_value()) {
                  auto nf = interpret_formula(*nf_);
                  if(nf.has_value()) {
                    return make_fptr(LatticeOrderPredicate<TermPtr, FormulaPtr>(std::move(*term), std::move(*u), std::move(*nf)));
                  }
                }
              }
              else {
                return make_fptr(LatticeOrderPredicate<TermPtr>(std::move(*term), std::move(*u)));
              }
            }
          }
      }
    }
    // Negative literal
    else if(f.is(F::Seq) && f.seq().size() == 1 && f.sig() == NOT &&
      (f.seq(0).is(F::V) || f.seq(0).is(F::LV)))
    {
      return interpret_literal<true>(f.seq(0));
    }
    // Positive literal
    else if(f.is(F::V) || f.is(F::LV)) {
      return interpret_literal<false>(f);
    }
    // Logical negation
    else if(f.is(F::Seq) && f.seq().size() == 1 && f.sig() == NOT) {
      return interpret_negation(f, neg_context);
    }
    // Singleton formula
    else if(f.is(F::Seq) && f.seq().size() == 1) {
      return interpret_formula(f.seq(0), neg_context);
    }
    return {};
  }

public:
  struct TellType {
    thrust::optional<typename A::TellType> sub;
    battery::vector<FormulaPtr, Allocator> props;
    TellType(TellType&&) = default;
    TellType& operator=(TellType&&) = default;
    TellType(A::TellType&& sub, const Allocator& alloc): sub(std::move(sub)), props(alloc) {}
    TellType(size_t n, const Allocator& alloc): sub(), props(alloc) {
      props.reserve(n);
    }
  };

  /** IPC expects a conjunction of the form \f$ c_1 \land \ldots \land c_n \f$ where sub-formulas \f$ c_i \f$ can either be interpreted in the sub-domain `A` or in the current domain.
    Moreover, we only treat exact conjunction (no under- or over-approximation of the conjunction).
    For now, \f$ T \neq k \f$ is not supported where \f$ T \f$ is an arithmetic term, containing function symbols supported in `terms.hpp`. */
  template <class F>
  CUDA thrust::optional<TellType> interpret(const F& f) {
    // If the formula is untyped, we first try to interpret it in the sub-domain.
    if(f.type() == UNTYPED || f.type() != uid) {
      auto r = a->interpret(f);
      if(r.has_value()) {
        return TellType(std::move(*r), props.get_allocator());
      }
      if(f.type() != UNTYPED) {
        return {}; // Reason: The formula `f` could not be interpreted in the sub-domain `A`.
      }
    }
    if(f.type() == UNTYPED || f.type() == uid) {
      // Conjunction
      if(f.is(F::Seq) && f.sig() == AND) {
        auto split_formula = extract_ty(f, uid);
        const auto& sub_ipc_ = battery::get<0>(split_formula);
        const auto& sub_ipc = sub_ipc_.seq();
        const auto& sub_a = battery::get<1>(split_formula);
        TellType res(sub_ipc.size(), props.get_allocator());
        // We need to interpret the formulas in the sub-domain first because it might handle existential quantifiers needed by formulas interpreted in this domain.
        if(sub_a.seq().size() > 0) {
          auto sub = a->interpret(sub_a);
          if(sub.has_value()) {
            res.sub = std::move(sub);
          }
          else {
            return {}; // Reason: A formula could not be interpreted in the sub-domain `A`.
          }
        }
        for(int i = 0; i < sub_ipc.size(); ++i) {
          auto prop = interpret_formula(sub_ipc[i]);
          if(prop.has_value()) {
            res.props.push_back(std::move(*prop));
          }
          else {
            return {}; // Reason: A formula could not be interpreted as a propagator.
          }
        }
        return std::move(res);
      }
      else {
        auto prop = interpret_formula(f);
        if(prop.has_value()) {
          TellType res(1, props.get_allocator());
          res.props.push_back(std::move(*prop));
          return std::move(res);
        }
        return {}; // Reason: The formula `f` could not be interpreted as a propagator.
      }
    }
    return {};
  }

  /** Note that we cannot add propagators in parallel (but modifying the underlying domain is ok).
      This is a current limitation that we should fix later on.
      Notes for later:
        * To implement "telling of propagators", we would need to check if a propagator has already been added or not (for idempotency).
        * 1. Walk through the existing propagators to check which ones are already in.
        * 2. If a propagator has the same shape but different constant `U`, join them in place.
        * 3. See paper notes for a technique to deal with copy. */
  CUDA this_type& tell(TellType&& t, BInc& has_changed) {
    if(t.sub.has_value()) {
      a->tell(*t.sub, has_changed);
    }
    if(t.props.size() > 0) {
      has_changed = BInc::top();
    }
    size_t n = props.size();
    props.reserve(n + t.props.size());
    for(int i = 0; i < t.props.size(); ++i) {
      props.push_back(std::move(t.props[i]));
      props[n + i]->preprocess(*a, has_changed);
    }
    return *this;
  }

  CUDA BInc ask(const TellType& t) const {
    for(int i = 0; t.props.size(); ++i) {
      if(!t.props[i]->ask(*a).guard()) {
        return BInc::bot();
      }
    }
    if(t.sub.has_value()) {
      return a->ask(*(t.sub));
    }
    return BInc::top();
  }

  CUDA size_t num_refinements() const {
    return props.size();
  }

  CUDA void refine(size_t i, BInc& has_changed) {
    assert(i < num_refinements());
    props[i]->refine(*a, has_changed);
  }

  CUDA void refine(BInc& has_changed) {
    for(size_t i = 0; i < props.size(); ++i) {
      refine(i, has_changed);
    }
  }

#ifdef __NVCC__

  DEVICE void refine(BInc& has_changed, gpu_tag_t) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    __shared__ BInc changed[3] = {BInc::bot(), BInc::bot(), BInc::bot()};
    for(int i = 1; a->is_top().guard() && changed[(i-1)%3]; ++i) {
      for (int t = tid; t < num_refinements(); t += stride) {
        refine(t, changed[i%3]);
      }
      changed[(i+1)%3].dtell(BInc::bot());
      has_changed.tell(changed[i%3]);
      __syncthreads();
    }
  }

#endif

  // Functions forwarded to the sub-domain `A`.

  /** `true` if the underlying abstract element is top, `false` otherwise. */
  CUDA BInc is_top() const {
    return a->is_top();
  }

  /** `true` if the underlying abstract element is bot and there is no refinement function, `false` otherwise. */
  CUDA BInc is_bot() const {
    return BInc(a->is_bot().guard() && props.size() == 0);
  }

  CUDA const Universe& project(AVar x) const {
    return a->project(x);
  }

  CUDA const Env& environment() const {
    return a->environment();
  }

  CUDA ZPInc<int> vars() const {
    return a->size();
  }
};

}

#endif
