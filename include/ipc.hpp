// Copyright 2021 Pierre Talbot

#ifndef IPC_HPP
#define IPC_HPP

#include "ast.hpp"
#include "terms.hpp"
#include "z.hpp"

namespace lala {

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

private:
  struct Propagator {
    Term<A>* left;
    Universe right;
    Propagator(): left(nullptr), right(Universe::bot()) {}
    Propagator(const Propagator&) = default;
    Propagator(Propagator&&) = default;
    Propagator& operator=(Propagator&&) = default;
    Propagator& operator=(const Propagator&) = default;
    Propagator(Term<A>* left, Universe right):
      left(left), right(right) {}
  };

  AType uid;
  A* a;
  DArray<Propagator, Allocator> props;

public:
  CUDA IPC(AType uid, A* a,
    Allocator alloc = Allocator()): a(a), props(alloc), uid(uid)  {}

  CUDA IPC(IPC&& other): props(std::move(other.props)), uid(other.uid) {
    swap(a, other.a);
  }

  CUDA static this_type bot(AType uid = UNTYPED) {
    Allocator allocator;
    A* a = new(allocator) A::bot();
    return IPC(uid, a);
  }

  /** A special symbolic element representing top. */
  CUDA static this_type top(AType uid = UNTYPED) {
    Allocator allocator;
    A* a = new(allocator) A::top();
    return IPC(uid, a);
  }

private:
  CUDA thrust::optional<Term<A>*> interpret_unary(Sig sig, Term<A>* a) {
    Allocator alloc = props.get_allocator();
    switch(sig) {
      case NEG:
        return new(alloc) DynTerm(Neg<Term<A>*>(std::move(a)));
      default:
        return {};
    }
  }

  CUDA thrust::optional<Term<A>*> interpret_binary(Sig sig,
    Term<A>* x, Term<A>* y)
  {
    Allocator alloc = props.get_allocator();
    switch(sig) {
      case ADD:
        return new(alloc) DynTerm(
          Add<Term<A>*, Term<A>*>(std::move(x), std::move(y)));
      case SUB:
        return new(alloc) DynTerm(
          Sub<Term<A>*, Term<A>*>(std::move(x), std::move(y)));
      case MUL:
        return new(alloc) DynTerm(
          Mul<Term<A>*, Term<A>*>(std::move(x), std::move(y)));
      case DIV:
        return new(alloc) DynTerm(
          Mul<Term<A>*, Term<A>*>(std::move(x), std::move(y)));
      default:
        return {};
    }
  }

  CUDA thrust::optional<Term<A>*> interpret_nary(Sig sig,
    DArray<Term<A>*, Allocator> subterms)
  {
    Allocator alloc = props.get_allocator();
    switch(sig) {
      case ADD:
        return new(alloc) DynTerm(
          NaryAdd<Term<A>*, Allocator>(std::move(subterms)));
      case MUL:
        return new(alloc) DynTerm(
          NaryMul<Term<A>*, Allocator>(std::move(subterms)));
      default:
        return {};
    }
  }

  template <class F>
  CUDA thrust::optional<Term<A>*> interpret_sequence(Sig sig,
    const typename F::Sequence& seq)
  {
    DArray<Term<A>*, Allocator> subterms(seq.size());
    for(int i = 0; i < subterms.size(); ++i) {
      auto t = interpret_term(seq[i]);
      if(!t.has_value()) {
        return {};
      }
      else {
        subterms[i] = std::move(*t);
      }
    }
    if(subterms.size() == 1) {
      Term<A>* x = subterms[0];
      subterms[0] = nullptr;
      return interpret_unary(sig, x);
    }
    else if(subterms.size() == 2) {
      // This code is not very good, it is necessary to set subterms[i] to nullptr, otherwsie DArray will free the pointers because DArray owns polymorphic objects.
      // I'm not sure about what would be a better solution at the moment.
      Term<A>* x = subterms[0];
      Term<A>* y = subterms[1];
      subterms[0] = nullptr;
      subterms[1] = nullptr;
      return interpret_binary(sig, x, y);
    }
    else {
      return interpret_nary(sig, std::move(subterms));
    }
  }

  template <class F>
  CUDA thrust::optional<Term<A>*> interpret_term(const F& f) {
    Allocator alloc = props.get_allocator();
    if(f.is(F::V)) {
      // need to check that this variable is defined in the domain?
      // or should we just suppose it is?
      return new(alloc) DynTerm(Variable<A>(f.v()));
    }
    else if(f.is(F::LV)) {
      auto v = a->environment().to_avar(f.lv());
      if(!v.has_value()) {
        return {};
      }
      else {
        return new(alloc) DynTerm(Variable<A>(*v));
      }
    }
    else if(f.is(F::Z)) {
      auto k = Universe::interpret(F::make_binary(F::make_avar(0), EQ, f));
      if(!k.has_value()) {
        return {};
      }
      else {
        return new(alloc) DynTerm(Constant<A>(std::move(*k)));
      }
    }
    else if(f.is(F::Seq)) {
      return interpret_sequence<F>(f.sig(), f.seq());
    }
    return {};
  }

  template <class F>
  CUDA thrust::optional<Propagator> interpret_one(const F& f) {
    assert(f.type() == uid);
    // Form of the constraint `T <op> u` with `x <op> u` interpreted in the underlying universe.
    if(f.is(F::Seq) && f.seq().size() == 2) {
      auto u = Universe::interpret(F::make_binary(F::make_avar(0), f.sig(), f.seq(1)));
      if(u.has_value()) {
        auto term = interpret_term(f.seq(0));
        if(term.has_value()) {
          return Propagator(*term, *u);
        }
      }
    }
    return {};
  }

public:
  struct TellType {
    thrust::optional<typename A::TellType> sub;
    DArray<Propagator, Allocator> props;
    TellType(TellType&&) = default;
    TellType& operator=(TellType&&) = default;
    TellType(A::TellType&& sub): sub(std::move(sub)), props() {}
    TellType(size_t n, const Allocator& alloc): sub(), props(n, alloc) {}
  };

  /** IPC expects a conjunction of the form \f$ \varphi \land c_1 \land \ldots \land c_n \f$ where \f$ \varphi \f$ can be interpret in the sub-domain `A` (and is possibly a conjunction), and \f$ c_i \f$ can be interpreted in the current domain.
    Moreover, we only treat exact conjunction (no under- or over-approximation of the conjunction).
    Each constraint \f$ c_i \f$ must be of the form \f$ T \leq k \f$, \f$ T < k \f$, \f$ T \geq k \f$, \f$ T > k \f$ or \f$ T = k \f$.
    For now, \f$ T \neq k \f$ is not supported.
    \f$ T \f$ is an arithmetic term, containing function symbols supported in `terms.hpp`. */
  template <class F>
  CUDA thrust::optional<TellType> interpret(const F& f) {
    if(f.type() != uid) {
      auto r = a->interpret(f);
      if(r.has_value()) {
        return TellType(std::move(*r));
      }
      return {}; // Reason: The formula `f` could not be interpreted in the sub-domain `A`.
    }
    // Conjunction
    else if(f.is(F::Seq) && f.sig() == AND) {
      const typename F::Sequence& seq = f.seq();
      size_t num_props = 0;
      size_t num_sub = 0;
      for(int i = 0; i < seq.size(); ++i) {
        if(seq[i].type() == uid) {
          num_props++;
        }
        else {
          num_sub++;
        }
      }
      if(num_sub > 1) {
        return {}; // Reason: More than one formula that needs to be interpreted in the sub-domain. They must be grouped together in a conjunction.
      }
      TellType res(num_props, props.get_allocator());
      for(int i = 0, p = 0; i < seq.size(); ++i) {
        if(seq[i].type() == uid) {
          auto prop = interpret_one(seq[i]);
          if(prop.has_value()) {
            res.props[p] = std::move(*prop);
            ++p;
          }
          else {
            return {}; // Reason: A formula could not be interpreted as a propagator.
          }
        }
        else {
          auto sub = a->interpret(seq[i]);
          if(sub.has_value()) {
            res.sub = std::move(sub);
          }
          else {
            return {}; // Reason: A formula could not be interpreted in the sub-domain `A`.
          }
        }
      }
      return std::move(res);
    }
    else {
      auto prop = interpret_one(f);
      if(prop.has_value()) {
        TellType res(1, props.get_allocator());
        res.props[0] = std::move(*prop);
        return std::move(res);
      }
      return {}; // Reason: The formula `f` could not be interpreted as a propagator.
    }
  }

  CUDA void resize(size_t new_size) {
    if(new_size > props.size()) {
      using Array = DArray<Propagator, Allocator>;
      Array props2 = Array(new_size);
      for(int i = 0; i < props.size(); ++i) {
        props2[i] = std::move(props[i]);
      }
      props = props2;
    }
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
    // !! Not correct, need to fill the temporary array first.
    if(t.props.size() > 0) {
      has_changed = BInc::top();
    }
    size_t n = props.size();
    resize(n + t.props.size());
    for(int i = 0; i < t.props.size(); ++i) {
      props[n + i] = std::move(t.props[i]);
      props[n + i].right.tell(props[n + i].left->project(*a), has_changed);
    }
    return *this;
  }

  CUDA size_t num_refinements() const {
    return props.size();
  }

  CUDA void refine(size_t i, BInc& has_changed) {
    assert(i < num_refinements());
    props[i].left->tell(*a, props[i].right, has_changed);
  }

  CUDA void refine(BInc& has_changed) {
    for(size_t i = 0; i < props.size(); ++i) {
      refine(i, has_changed);
    }
  }

  // Functions forwarded to the sub-domain `A`.

  /** `true` if the underlying abstract element is top, `false` otherwise. */
  CUDA BInc is_top() const {
    return a->is_top();
  }

  /** `true` if the underlying abstract element is bot, `false` otherwise. */
  CUDA BInc is_bot() const {
    return a->is_bot() && props.size() == 0;
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
