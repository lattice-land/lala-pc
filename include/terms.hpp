// Copyright 2021 Pierre Talbot

#ifndef TERMS_HPP
#define TERMS_HPP

#include "arithmetic.hpp"

namespace lala {

// NOTE: Bitwise OR and AND are necessary to avoid short-circuit of Boolean operators.
// NOTE: The pointers are shared among terms.
//       This is because terms have a constant state, that is, terms are read-only once constructed.
//       Therefore, the copy constructor does not perform a deep copy.

template <typename AD>
class Term {
public:
  using A = AD;
  using U = typename A::Universe;
  CUDA virtual ~Term() {}
  CUDA virtual void tell(A&, const U&, BInc&) const = 0;
  CUDA virtual U project(A&) const = 0;
  CUDA virtual BInc is_top(const A&) const = 0;
  CUDA virtual void print(const A&) const = 0;
};

// `DynTerm` wraps a term and inherits from Term.
// A vtable will be created.
template<typename BaseTerm>
class DynTerm: public Term<typename BaseTerm::A> {
  BaseTerm t;
public:
  using A = typename BaseTerm::A;
  using U = typename A::Universe;

  CUDA DynTerm(const BaseTerm& t): t(t) {}
  CUDA DynTerm(const DynTerm<BaseTerm>& other): DynTerm(other.t) {}

  CUDA void tell(A& a, const U& u, BInc& has_changed) const override {
    t.tell(a, u, has_changed);
  }

  CUDA U project(A& a) const override {
    t.project(a);
  }

  CUDA BInc is_top(const A& a) const override {
    return t.is_top(a);
  }

  CUDA void print(const A& a) const override {
    t.print(a);
  }

  CUDA ~DynTerm() {}
};

// This function is used to dereference the attribute if T is a pointer.
// The rational behind that, is to be able to manipulate a type T as a pointer or a reference.
// In the following code, our term AST is either static (only with template) or dynamic (with virtual function call).
// But we did not want to duplicate the code to handle both.
template <typename T>
CUDA const typename std::remove_pointer<T>::type& deref(const T& x) {
  if constexpr(std::is_pointer<T>()) {
    return *x;
  }
  else {
    return x;
  }
  return 0; // unreachable (to avoid a compiler warning)
}

template <typename AD>
class Constant {
public:
  using A = AD;
  using U = typename A::Universe;

private:
  U k;

public:
  CUDA Constant(U k) : k(k) {}
  CUDA Constant(const Constant<A>& other): k(other.k) {}
  CUDA void tell(A&, const U&, BInc&) const {}
  CUDA U project(A&) const { return k; }
  CUDA BInc is_top(const A&) const { return BInc::bot(); }
  CUDA void print(const A&) const { ::print(k); }
};

template <typename AD>
class Variable {
public:
  using A = AD;
  using U = typename A::Universe;

private:
  AVar avar;

public:
  CUDA Variable(AVar avar) : avar(avar) {}
  CUDA Variable(const Variable<A>& other): Variable(other.avar) {}

  CUDA void tell(A& a, const U& u, BInc& has_changed) const {
    a.tell(avar, u, has_changed);
  }

  CUDA U project(const A& a) const {
    return a.project(avar);
  }

  CUDA BInc is_top(const A& a) const { return project(a).is_top(); }
  CUDA void print(const A& a) const { a.environment().to_lvar(avar).print(); }
};

template<class U, Approx appx = EXACT>
struct GroupAdd {
  CUDA static U op(const U& a, const U& b) {
    return add<appx>(a, b);
  }

  CUDA static U inv(const U& a, const U& b) {
    return sub<appx>(a, b);
  }

  CUDA static char symbol() { return '+'; }
};

template<class U, Approx appx = OVER>
struct GroupMul {
  CUDA static U op(const U& a, const U& b) {
    return mul<appx>(a, b);
  }

  CUDA static U inv(const U& a, const U& b) {
    return div<appx>(a, b);
  }

  CUDA static char symbol() { return '*'; }
};

template <class Group, class TermX, class TermY>
class Binary {
public:
  using A = typename TermX::A;
  using U = typename TermX::U;
  using G = Group;
  using this_type = Binary<Group, TermX, TermY>;

private:
  TermX x_term;
  TermY y_term;

  using TermX_ = typename std::remove_pointer<TermX>::type;
  using TermY_ = typename std::remove_pointer<TermY>::type;

  CUDA INLINE const TermX_& x() const {
    return deref(x_term);
  }

  CUDA INLINE const TermY_& y() const {
    return deref(y_term);
  }

public:
  CUDA Binary(const TermX& x_term, const TermY& y_term): x_term(x_term), y_term(y_term) {}
  CUDA Binary(const this_type& other): Binary(other.x_term, other.y_term) {}

  /** Enforce `x + y >= u` where >= is the lattice order of the universe of discourse.
      For instance, over the interval abstract universe, `x + y >= [2..5]` will ensure that `x + y` is eventually at least `2` and at most `5`. */
  CUDA void tell(A& a, const U& u, BInc& has_changed) const {
    auto xt = x().project(a);
    auto yt = y().project(a);
    if(lor(xt.is_top(), yt.is_top()).guard()) { return; }
    if constexpr(!std::is_same_v<TermX, Constant>()) {
      x().tell(a, G::inv(u, y().project(a)), has_changed);   // x <- u - y
    }
    if constexpr(!std::is_same_v<TermY, Constant>()) {
      y().tell(a, G::inv(u, x().project(a)), has_changed);   // y <- u - x
    }
  }

  CUDA U project(const A& a) const {
    return G::op(x().project(), y().project());
  }

  CUDA BInc is_top(const A& a) const {
    return lor(x().is_top(a), y().is_top(a));
  }

  CUDA void print(const A& a) const {
    x().print(a);
    printf(" %c ", G::symbol());
    y().print(a);
  }
};

template <class TermX, class TermY, Approx appx = EXACT>
using Add = Binary<
  GroupAdd<typename std::remove_pointer<TermX>::type::U, appx>,
  TermX,
  TermY>;

template <class TermX, class TermY, Approx appx = OVER>
using Mul = Binary<
  GroupMul<typename std::remove_pointer<TermX>::type::U, appx>,
  TermX,
  TermY>;

template <class T, class Combinator, class Allocator>
class Nary {
  DArray<T, Allocator> terms;

  using T_ = typename std::remove_pointer<T>::type;
  CUDA INLINE const T_& t(size_t i) const {
    return deref(terms[i]);
  }

public:
  using this_type = Nary<T, Combinator, Allocator>;
  using A = typename Combinator::A;
  using U = typename Combinator::U;
  using G = typename Combinator::G;

  CUDA Nary(DArray<T, Allocator>&& terms): terms(terms) {}
  CUDA Nary(const this_type& other): Nary(other.terms) {}

  CUDA U project(const A& a) const {
    U accu = t(0).project(a);
    for(int i = 1; i < terms.size(); ++i) {
      accu = G::op(accu, t(i).project(a));
    }
    return accu;
  }

  CUDA void tell(A& a, const U& u, BInc& has_changed) const {
    U all = project(a);
    for(int i = 0; i < terms.size(); ++i) {
      t(i).tell(a, G::inv(u, G::inv(all, t(i).project(a))), has_changed);
    }
  }

  CUDA BInc is_top(const A& a) const {
    for(int i = 0; i < terms.size(); ++i) {
      if(t(i).is_top().guard()) {
        return BInc::top();
      }
    }
    return BInc::bot();
  }

  CUDA void print(const A& a) const {
    t(0).print(a);
    for(int i = 1; i < terms.size(); ++i) {
      printf(" %c ", G::symbol());
      t(i).print(a);
    }
  }
};

template<class T, class Allocator, Approx appx = EXACT>
using NaryAdd = Nary<T, Add<T, T, appx>, Allocator>;

template<class T, class Allocator, Approx appx = OVER>
using NaryMul = Nary<T, Mul<T, T, appx>, Allocator>;

} // namespace lala

#endif
