// Copyright 2021 Pierre Talbot

#ifndef LALA_PC_FORMULA_HPP
#define LALA_PC_FORMULA_HPP

#include "lala/universes/arith_bound.hpp"

namespace lala {
namespace pc {

template <class AD, class Allocator>
class Formula;

template <class AD, class Allocator>
class AbstractElement {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;

  using tell_type = typename A::template tell_type<allocator_type>;
  using ask_type = typename A::template ask_type<allocator_type>;
  using this_type = AbstractElement<A, allocator_type>;

  template <class A2, class Alloc>
  friend class AbstractElement;

private:
  tell_type tellv;
  tell_type not_tellv;
  ask_type askv;
  ask_type not_askv;

public:
  CUDA AbstractElement() {}

  CUDA AbstractElement(
    tell_type&& tellv, tell_type&& not_tellv,
    ask_type&& askv, ask_type&& not_askv)
   : tellv(std::move(tellv)), not_tellv(std::move(not_tellv)), askv(std::move(askv)), not_askv(std::move(not_askv)) {}

  AbstractElement(this_type&& other) = default;
  this_type& operator=(this_type&& other) = default;

  template <class A2, class Alloc2>
  CUDA AbstractElement(const AbstractElement<A2, Alloc2>& other, const allocator_type& alloc)
   : tellv(other.tellv, alloc), not_tellv(other.not_tellv, alloc)
   , askv(other.askv, alloc), not_askv(other.not_askv, alloc) {}

public:
  CUDA local::B ask(const A& a) const {
    return a.ask(askv);
  }

  CUDA local::B nask(const A& a) const {
    return a.ask(not_askv);
  }

  CUDA bool deduce(A& a) const {
    return a.deduce(tellv);
  }

  CUDA bool contradeduce(A& a) const {
    return a.deduce(not_tellv);
  }

  CUDA NI void print(const A& a) const {
    printf("<abstract element (%d)>", a.aty());
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType, Allocator2 allocator = Allocator2()) const {
    return a.deinterpret(tellv, env, allocator);
  }

  CUDA size_t length() const { return 1; }
};

/** A Boolean variable defined on a universe of discourse supporting `0` for false and `1` for true. */
template<class AD, bool neg>
class VariableLiteral {
public:
  using A = AD;
  using U = typename A::local_universe;

  template <class A2, bool neg2>
  friend class VariableLiteral;

private:
  AVar avar;

public:
  VariableLiteral() = default;
  CUDA VariableLiteral(AVar avar): avar(avar) {}

  template <class A2, class Alloc>
  CUDA VariableLiteral(const VariableLiteral<A2, neg>& other, const Alloc&): avar(other.avar) {}

private:
  template<bool negate>
  CUDA local::B ask_impl(const A& a) const {
    U tmp{};
    a.project(avar, tmp);
    if constexpr(negate) {
      return tmp <= U::eq_zero();
    }
    else {
      return !(tmp >= U::eq_zero());
    }
  }

  template <bool negate>
  CUDA bool deduce_impl(A& a) const {
    if constexpr(negate) {
      return a.embed(avar, U::eq_zero());
    }
    else {
      return a.embed(avar, U::eq_one());
    }
  }

public:
  /** Given a variable `x` taking a value in a universe `U` denoted by \f$ a(x) \f$.
   *   - \f$ a \vDash x \f$ holds iff \f$ \lnot (a(x) \leq [\![x = 0]\!]_U) \f$.
   *   - \f$ a \vDash \lnot{x} \f$ holds iff \f$ a(x) \geq [\![x = 0]\!]_U \f$. */
  CUDA local::B ask(const A& a) const {
    return ask_impl<neg>(a);
  }

  /** Given a variable `x` taking a value in a universe `U` denoted by \f$ a(x) \f$.
   *   - \f$ a \nvDash x \f$ holds iff \f$ a(x) \geq [\![x = 0]\!]_U \f$.
   *   - \f$ a \nvDash \lnot{x} \f$ holds iff \f$ \lnot (a(x) \leq [\![x = 0]\!]_U) \f$. */
  CUDA local::B nask(const A& a) const {
    return ask_impl<!neg>(a);
  }

  /** Perform:
   *    * Positive literal: \f$ x = a(x) \sqcup [\![x = 1]\!]_U \f$
   *    * Negative literal: \f$ x = a(x) \sqcup [\![x = 0]\!]_U \f$. */
  CUDA bool deduce(A& a) const {
    return deduce_impl<neg>(a);
  }

  /** Perform:
   *    * Positive literal: \f$ x = a(x) \sqcup [\![x = 0]\!]_U \f$
   *    * Negative literal: \f$ x = a(x) \sqcup [\![x = 1]\!]_U \f$. */
  CUDA bool contradeduce(A& a) const {
    return deduce_impl<!neg>(a);
  }

  CUDA NI void print(const A& a) const {
    if constexpr(neg) { printf("not "); }
    printf("(%d,%d)", avar.aty(), avar.vid());
  }

  template <class Env, class Allocator = typename Env::allocator_type>
  CUDA NI TFormula<Allocator> deinterpret(const A&, const Env& env, AType apc, Allocator allocator = Allocator()) const {
    using F = TFormula<Allocator>;
    auto f = F::make_lvar(avar.aty(), LVar<Allocator>(env.name_of(avar), allocator));
    if constexpr(neg) {
      f = F::make_unary(NOT, f, apc);
    }
    return f;
  }

  CUDA size_t length() const { return 1; }
};

template<class AD>
class False {
public:
  using A = AD;
  using U = typename A::local_universe;

  template <class A2>
  friend class False;

public:
  False() = default;

  template <class A2, class Alloc>
  CUDA False(const False<A2>&, const Alloc&) {}

  CUDA local::B ask(const A& a) const { return a.is_bot(); }
  CUDA local::B nask(const A&) const { return true; }

  CUDA bool deduce(A& a) const {
    if(!a.is_bot()) {
      a.meet_bot();
      return true;
    }
    return false;
  }

  CUDA bool contradeduce(A&) const { return false; }
  CUDA NI void print(const A& a) const { printf("false"); }

  template <class Env, class Allocator = typename Env::allocator_type>
  CUDA NI TFormula<Allocator> deinterpret(const A&, const Env&, AType, Allocator allocator = Allocator()) const {
    return TFormula<Allocator>::make_false();
  }

  CUDA size_t length() const { return 1; }
};

template<class AD>
class True {
public:
  using A = AD;
  using U = typename A::local_universe;

  template <class A2>
  friend class True;

public:
  True() = default;

  template <class A2, class Alloc>
  CUDA True(const True<A2>&, const Alloc&) {}

  CUDA local::B ask(const A&) const { return true; }
  CUDA local::B nask(const A& a) const { return a.is_bot(); }
  CUDA bool deduce(A&) const { return false; }
  CUDA bool contradeduce(A& a) const {
    if(!a.is_bot()) {
      a.meet_bot();
      return true;
    }
    return false;
  }
  CUDA void print(const A& a) const { printf("true"); }

  template <class Env, class Allocator = typename Env::allocator_type>
  CUDA TFormula<Allocator> deinterpret(const A&, const Env&, AType, Allocator allocator = Allocator()) const {
    return TFormula<Allocator>::make_true();
  }

  CUDA size_t length() const { return 1; }
};

template<class AD, class Allocator>
class Conjunction {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Conjunction<A, allocator_type>;
  using sub_type = Formula<A, allocator_type>;
  using sub_ptr = battery::unique_ptr<sub_type, allocator_type>;

  template <class A2, class Alloc2>
  friend class Conjunction;

private:
  sub_ptr f;
  sub_ptr g;

public:
  CUDA Conjunction(sub_ptr&& f, sub_ptr&& g): f(std::move(f)), g(std::move(g)) {}
  CUDA Conjunction(this_type&& other): Conjunction(std::move(other.f), std::move(other.g)) {}

  template <class A2, class Alloc2>
  CUDA Conjunction(const Conjunction<A2, Alloc2>& other, const allocator_type& alloc)
   : f(battery::allocate_unique<sub_type>(alloc, *other.f, alloc))
   , g(battery::allocate_unique<sub_type>(alloc, *other.g, alloc))
  {}

  CUDA local::B ask(const A& a) const {
    return f->ask(a) && g->ask(a);
  }

  CUDA local::B nask(const A& a) const {
    return f->nask(a) || g->nask(a);
  }

  CUDA bool deduce(A& a) const {
    bool has_changed = f->deduce(a);
    has_changed |= g->deduce(a);
    return has_changed;
  }

  CUDA bool contradeduce(A& a) const {
    if(f->ask(a)) { return g->contradeduce(a); }
    else if(g->ask(a)) { return f->contradeduce(a); }
    return false;
  }

  CUDA NI void print(const A& a) const {
    f->print(a);
    printf(" /\\ ");
    g->print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    auto left = f->deinterpret(a, env, apc, allocator);
    auto right = g->deinterpret(a, env, apc, allocator);
    if(left.is_true()) { return right; }
    else if(right.is_true()) { return left; }
    else if(left.is_false()) { return left; }
    else if(right.is_false()) { return right; }
    else {
      return TFormula<Allocator2>::make_binary(left, AND, right, apc, allocator);
    }
  }

  CUDA size_t length() const { return 1 + f->length() + g->length(); }
};

template<class AD, class Allocator>
class Disjunction {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Disjunction<A, allocator_type>;
  using sub_type = Formula<A, allocator_type>;
  using sub_ptr = battery::unique_ptr<sub_type, allocator_type>;

  template <class A2, class Alloc2>
  friend class Disjunction;

private:
  sub_ptr f;
  sub_ptr g;

public:
  CUDA Disjunction(sub_ptr&& f, sub_ptr&& g): f(std::move(f)), g(std::move(g)) {}
  CUDA Disjunction(this_type&& other): Disjunction(
    std::move(other.f), std::move(other.g)) {}

  template <class A2, class Alloc2>
  CUDA Disjunction(const Disjunction<A2, Alloc2>& other, const allocator_type& alloc)
   : f(battery::allocate_unique<sub_type>(alloc, *other.f, alloc))
   , g(battery::allocate_unique<sub_type>(alloc, *other.g, alloc))
  {}

  CUDA local::B ask(const A& a) const {
    return f->ask(a) || g->ask(a);
  }

  CUDA local::B nask(const A& a) const {
    return f->nask(a) && g->nask(a);
  }

  CUDA bool deduce(A& a) const {
    if(f->nask(a)) { return g->deduce(a); }
    else if(g->nask(a)) { return f->deduce(a); }
    return false;
  }

  CUDA bool contradeduce(A& a) const {
    bool has_changed = f->contradeduce(a);
    has_changed |= g->contradeduce(a);
    return has_changed;
  }

  CUDA NI void print(const A& a) const {
    f->print(a);
    printf(" \\/ ");
    g->print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    auto left = f->deinterpret(a, env, apc, allocator);
    auto right = g->deinterpret(a, env, apc, allocator);
    if(left.is_true()) { return left; }
    else if(right.is_true()) { return right; }
    else if(left.is_false()) { return right; }
    else if(right.is_false()) { return left; }
    else {
      return TFormula<Allocator2>::make_binary(left, OR, right, apc, allocator);
    }
  }

  CUDA size_t length() const { return 1 + f->length() + g->length(); }
};

template<class AD, class Allocator>
class Biconditional {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Biconditional<AD, Allocator>;
  using sub_type = Formula<A, allocator_type>;
  using sub_ptr = battery::unique_ptr<sub_type, allocator_type>;

  template <class A2, class Alloc2>
  friend class Biconditional;

private:
  sub_ptr f;
  sub_ptr g;

public:
  CUDA Biconditional(sub_ptr&& f, sub_ptr&& g): f(std::move(f)), g(std::move(g)) {}
  CUDA Biconditional(this_type&& other): Biconditional(
    std::move(other.f), std::move(other.g)) {}

  template <class A2, class Alloc2>
  CUDA Biconditional(const Biconditional<A2, Alloc2>& other, const allocator_type& alloc)
   : f(battery::allocate_unique<sub_type>(alloc, *other.f, alloc))
   , g(battery::allocate_unique<sub_type>(alloc, *other.g, alloc))
  {}

  CUDA local::B ask(const A& a) const {
    return
      (f->ask(a) && g->ask(a)) ||
      (f->nask(a) && g->nask(a));
  }

  // note that not(f <=> g) is equivalent to (f <=> not g)
  CUDA local::B nask(const A& a) const {
    return
      (f->ask(a) && g->nask(a)) ||
      (f->nask(a) && g->ask(a));
  }

  CUDA bool deduce(A& a) const {
    if(f->ask(a)) { return g->deduce(a); }
    else if(f->nask(a)) { return g->contradeduce(a); }
    else if(g->ask(a)) { return f->deduce(a); }
    else if(g->nask(a)) { return f->contradeduce(a); }
    return false;
  }

  // note that not(f <=> g) is equivalent to (f <=> not g)
  CUDA bool contradeduce(A& a) const {
    if(f->ask(a)) { return g->contradeduce(a); }
    else if(f->nask(a)) { return g->deduce(a); }
    else if(g->ask(a)) { return f->contradeduce(a); }
    else if(g->nask(a)) { return f->deduce(a); }
    return false;
  }

  CUDA NI void print(const A& a) const {
    f->print(a);
    printf(" <=> ");
    g->print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    return TFormula<Allocator2>::make_binary(
      f->deinterpret(a, env, apc, allocator), EQUIV, g->deinterpret(a, env, apc, allocator), apc, allocator);
  }

  CUDA size_t length() const { return 1 + f->length() + g->length(); }
};

template<class AD, class Allocator>
class Implication {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Implication<AD, Allocator>;
  using sub_type = Formula<A, allocator_type>;
  using sub_ptr = battery::unique_ptr<sub_type, allocator_type>;

  template <class A2, class Alloc2>
  friend class Implication;

private:
  sub_ptr f;
  sub_ptr g;

public:
  CUDA Implication(sub_ptr&& f, sub_ptr&& g): f(std::move(f)), g(std::move(g)) {}
  CUDA Implication(this_type&& other): Implication(
    std::move(other.f), std::move(other.g)) {}

  template <class A2, class Alloc2>
  CUDA Implication(const Implication<A2, Alloc2>& other, const allocator_type& alloc)
   : f(battery::allocate_unique<sub_type>(alloc, *other.f, alloc))
   , g(battery::allocate_unique<sub_type>(alloc, *other.g, alloc))
  {}

  // note that f => g is equivalent to (not f) or g.
  CUDA local::B ask(const A& a) const {
    return f->nask(a) || g->ask(a);
  }

  // note that not(f => g) is equivalent to f and (not g)
  CUDA local::B nask(const A& a) const {
    return f->ask(a) && g->nask(a);
  }

  CUDA bool deduce(A& a) const {
    if(f->ask(a)) { return g->deduce(a); }
    else if(g->nask(a)) { return f->contradeduce(a); }
    return false;
  }

  CUDA bool contradeduce(A& a) const {
    if(f->ask(a)) { return g->contradeduce(a); }
    else if(g->nask(a)) { return f->deduce(a); }
    return false;
  }

  CUDA NI void print(const A& a) const {
    f->print(a);
    printf(" => ");
    g->print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    return TFormula<Allocator2>::make_binary(
      f->deinterpret(a, env, apc, allocator), IMPLY, g->deinterpret(a, env, apc, allocator), apc, allocator);
  }

  CUDA size_t length() const { return 1 + f->length() + g->length(); }
};

template<class AD, class Allocator>
class ExclusiveDisjunction {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = ExclusiveDisjunction<AD, Allocator>;
  using sub_type = Formula<A, allocator_type>;
  using sub_ptr = battery::unique_ptr<sub_type, allocator_type>;

  template <class A2, class Alloc2>
  friend class ExclusiveDisjunction;

private:
  sub_ptr f;
  sub_ptr g;

public:
  CUDA ExclusiveDisjunction(sub_ptr&& f, sub_ptr&& g): f(std::move(f)), g(std::move(g)) {}
  CUDA ExclusiveDisjunction(this_type&& other): ExclusiveDisjunction(
    std::move(other.f), std::move(other.g)) {}

  template <class A2, class Alloc2>
  CUDA ExclusiveDisjunction(const ExclusiveDisjunction<A2, Alloc2>& other, const allocator_type& alloc)
   : f(battery::allocate_unique<sub_type>(alloc, *other.f, alloc))
   , g(battery::allocate_unique<sub_type>(alloc, *other.g, alloc))
  {}

  CUDA local::B ask(const A& a) const {
    return
      (f->ask(a) && g->nask(a)) ||
      (f->nask(a) && g->ask(a));
  }

  CUDA local::B nask(const A& a) const {
    return
      (f->ask(a) && g->ask(a)) ||
      (f->ask(a) && g->nask(a));
  }

  CUDA bool deduce(A& a) const {
    if(f->ask(a)) { return g->contradeduce(a); }
    else if(f->nask(a)) { return g->deduce(a); }
    else if(g->ask(a)) { return f->contradeduce(a); }
    else if(g->nask(a)) { return f->deduce(a); }
    return false;
  }

  CUDA bool contradeduce(A& a) const {
    if(f->ask(a)) { return g->deduce(a); }
    else if(f->nask(a)) { return g->contradeduce(a); }
    else if(g->ask(a)) { return f->deduce(a); }
    else if(g->nask(a)) { return f->contradeduce(a); }
    return false;
  }

  CUDA NI void print(const A& a) const {
    f->print(a);
    printf(" xor ");
    g->print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    return TFormula<Allocator2>::make_binary(
      f->deinterpret(a, env, apc, allocator), XOR, g->deinterpret(a, env, apc, allocator), apc, allocator);
  }

  CUDA size_t length() const { return 1 + f->length() + g->length(); }
};

template<class AD, class Allocator, bool neg = false>
class Equality {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Equality<A, allocator_type, neg>;
  using sub_type = Term<A, allocator_type>;

  template <class A2, class Alloc2, bool neg2>
  friend class Equality;

private:
  using LB = typename U::LB::local_type;
  using UB = typename U::UB::local_type;

  sub_type left;
  sub_type right;

public:
  CUDA Equality(sub_type&& left, sub_type&& right): left(std::move(left)), right(std::move(right)) {}
  CUDA Equality(this_type&& other): Equality(std::move(other.left), std::move(other.right)) {}

  template <class A2, class Alloc2>
  CUDA Equality(const Equality<A2, Alloc2, neg>& other, const allocator_type& alloc):
    left(other.left, alloc),
    right(other.right, alloc)
  {}

private:
  template <bool negate>
  CUDA local::B ask_impl(const A& a) const {
    U l{};
    U r{};
    left.project(a, l);
    right.project(a, r);
    if constexpr(negate) {
      return fmeet(l, r).is_bot();
    }
    else {
      return l == r && dual<UB>(l.lb()) == l.ub();
    }
  }

  template <bool negate>
  CUDA bool deduce_impl(A& a) const {
    U l{};
    U r{};
    bool has_changed = false;
    if constexpr(negate) {
      if(!right.is(sub_type::IConstant)) {
        left.project(a, l);
        if(dual<UB>(l.lb()) == l.ub()) {
          if constexpr(U::complemented) {
            return right.embed(a, l.complement());
          }
          else if (U::preserve_concrete_covers && U::is_arithmetic) {
            right.project(a, r);
            U lb{r};
            U ub{r};
            lb.meet_lb(LB::prev(l.lb()));
            ub.meet_ub(UB::prev(l.ub()));
            return right.embed(a, fjoin(lb, ub));
          }
        }
      }
      if(!left.is(sub_type::IConstant)) {
        right.project(a, r);
        if(dual<UB>(r.lb()) == r.ub()) {
          if constexpr(U::complemented) {
            return left.embed(a, r.complement());
          }
          else if (U::preserve_concrete_covers && U::is_arithmetic) {
            left.project(a, l);
            U lb{l};
            U ub{l};
            lb.meet_lb(LB::prev(r.lb()));
            ub.meet_ub(UB::prev(r.ub()));
            return left.embed(a, fjoin(lb, ub));
          }
        }
      }
    }
    else {
      if(!right.is(sub_type::IConstant)) {
        left.project(a, l);
        has_changed = right.embed(a, l);
      }
      if(!left.is(sub_type::IConstant)) {
        right.project(a, r);
        has_changed |= left.embed(a, r);
      }
    }
    return has_changed;
  }

public:
  CUDA local::B ask(const A& a) const {
    return ask_impl<neg>(a);
  }

  CUDA local::B nask(const A& a) const {
    return ask_impl<!neg>(a);
  }

  CUDA bool deduce(A& a) const {
    return deduce_impl<neg>(a);
  }

  CUDA bool contradeduce(A& a) const {
    return deduce_impl<!neg>(a);
  }

  CUDA NI void print(const A& a) const {
    left.print(a);
    if constexpr(neg) {
      printf(" != ");
    }
    else {
      printf(" == ");
    }
    right.print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    auto lf = left.deinterpret(a, env, apc, allocator);
    auto rf = right.deinterpret(a, env, apc, allocator);
    return TFormula<Allocator2>::make_binary(std::move(lf), neg ? NEQ : EQ, std::move(rf), apc, allocator);
  }

  CUDA size_t length() const { return 1 + left.length() + right.length(); }
};

template<class AD, class Allocator>
using Disequality = Equality<AD, Allocator, true>;

/** Implement the constraint `t1 <= t2` or `t1 > t2` if `neg` is `true`. */
template<class AD, class Allocator, bool neg>
class Inequality {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Inequality<A, allocator_type, neg>;
  using sub_type = Term<A, allocator_type>;

  template <class A2, class Alloc2, bool neg2>
  friend class Inequality;

private:
  using LB = typename U::LB::local_type;
  using UB = typename U::UB::local_type;

  sub_type left;
  sub_type right;

public:
  CUDA Inequality(sub_type&& left, sub_type&& right): left(std::move(left)), right(std::move(right)) {}
  CUDA Inequality(this_type&& other): Inequality(std::move(other.left), std::move(other.right)) {}

  template <class A2, class Alloc2>
  CUDA Inequality(const Inequality<A2, Alloc2, neg>& other, const allocator_type& alloc):
    left(other.left, alloc),
    right(other.right, alloc)
  {}

private:
  template <bool negate>
  CUDA local::B ask_impl(const A& a) const {
    U l{};
    U r{};
    left.project(a, l);
    right.project(a, r);
    if constexpr(negate) {
      return dual<UB>(l.lb()) > r.ub();
    }
    else {
      return l.ub() <= dual<UB>(r.lb());
    }
  }

  template <bool negate>
  CUDA bool deduce_impl(A& a) const {
    U l{};
    U r{};
    bool has_changed = false;
    // l > r
    if constexpr(negate) {
      if(!left.is(sub_type::IConstant)) {
        right.project(a, r);
        if constexpr(U::preserve_concrete_covers && U::is_arithmetic) {
          r.meet_lb(LB::prev(r.lb()));
        }
        has_changed = left.embed(a, U(r.lb(), UB::top()));
      }
      if(!right.is(sub_type::IConstant)) {
        left.project(a, l);
        if constexpr(U::preserve_concrete_covers && U::is_arithmetic) {
          l.meet_ub(UB::prev(l.ub()));
        }
        has_changed |= right.embed(a, U(LB::top(), l.ub()));
      }
    }
    // l <= r
    else {
      if(!left.is(sub_type::IConstant)) {
        right.project(a, r);
        has_changed |= left.embed(a, U(LB::top(), r.ub()));
      }
      if(!right.is(sub_type::IConstant)) {
        left.project(a, l);
        has_changed = right.embed(a, U(l.lb(), UB::top()));
      }
    }
    return has_changed;
  }

public:
  CUDA local::B ask(const A& a) const {
    return ask_impl<neg>(a);
  }

  CUDA local::B nask(const A& a) const {
    return ask_impl<!neg>(a);
  }

  CUDA bool deduce(A& a) const {
    return deduce_impl<neg>(a);
  }

  CUDA bool contradeduce(A& a) const {
    return deduce_impl<!neg>(a);
  }

  CUDA NI void print(const A& a) const {
    left.print(a);
    if constexpr(neg) {
      printf(" > ");
    }
    else {
      printf(" <= ");
    }
    right.print(a);
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    auto lf = left.deinterpret(a, env, apc, allocator);
    auto rf = right.deinterpret(a, env, apc, allocator);
    return TFormula<Allocator2>::make_binary(std::move(lf), neg ? GT : LEQ, std::move(rf), apc, allocator);
  }

  CUDA size_t length() const { return 1 + left.length() + right.length(); }
};

template<class AD, class Allocator>
using InequalityGT = Inequality<AD, Allocator, true>;

template<class AD, class Allocator>
using InequalityLEQ = Inequality<AD, Allocator, false>;

/**
 * A formula can occur in a term, e.g., `(x = 2) + (y = 2) + (z = 2) >= 2`
 * In that case, the entailment of the formula is mapped onto a sublattice of `U` supporting initialization from `0` and `1`.
 * A term can occur in a formula, as it is usually done.
 * Therefore, a formula is both a formula and a term.
 *
 * A logical formula is turned into a term by mapping their satisfiability to and from a sublattice of `U` representing Boolean values:
 *    - Consists of four distinct values \f$\{[\![x = 0]\!]_U \sqcap [\![x = 1]\!]_U, [\![x = 0]\!]_U, [\![x = 1]\!]_U, \top_U \}\f$.
 *    - With \f$\{[\![x = 0]\!]_U \sqcap [\![x = 1]\!]_U \f$ meaning neither true or false yet (e.g., unknown), \f$\{[\![x = 0]\!]_U \f$ modelling falsity, \f$ [\![x = 1]\!]_U \f$ modelling truth, and \f$ \top_U \f$ a logical statement both true and false (i.e., one of the variable is top).
*/
template <class AD, class Allocator>
class Formula {
public:
  using A = AD;
  using allocator_type = Allocator;
  using this_type = Formula<A, allocator_type>;
  using U = typename A::local_universe;
  using term_type = Term<A, allocator_type>;
  using this_ptr = battery::unique_ptr<this_type, allocator_type>;

  using PVarLit = VariableLiteral<A, false>;
  using NVarLit = VariableLiteral<A, true>;
  using Leq = InequalityLEQ<A, allocator_type>;
  using Gt = InequalityGT<A, allocator_type>;
  using Eq = Equality<A, allocator_type>;
  using Neq = Disequality<A, allocator_type>;
  using Conj = Conjunction<A, allocator_type>;
  using Disj = Disjunction<A, allocator_type>;
  using Bicond = Biconditional<A, allocator_type>;
  using Imply = Implication<A, allocator_type>;
  using Xor = ExclusiveDisjunction<A, allocator_type>;
  using AE = AbstractElement<A, allocator_type>;

  static constexpr size_t IPVarLit = 0;
  static constexpr size_t INVarLit = IPVarLit + 1;
  static constexpr size_t ITrue = INVarLit + 1;
  static constexpr size_t IFalse = ITrue + 1;
  static constexpr size_t ILeq = IFalse + 1;
  static constexpr size_t IGt = ILeq + 1;
  static constexpr size_t IEq = IGt + 1;
  static constexpr size_t INeq = IEq + 1;
  static constexpr size_t IConj = INeq + 1;
  static constexpr size_t IDisj = IConj + 1;
  static constexpr size_t IBicond = IDisj + 1;
  static constexpr size_t IImply = IBicond + 1;
  static constexpr size_t IXor = IImply + 1;
  static constexpr size_t IAE = IXor + 1;

  template <class A2, class Alloc2>
  friend class Formula;

private:
  using VFormula = battery::variant<
    PVarLit,
    NVarLit,
    True<A>,
    False<A>,
    Leq,
    Gt,
    Eq,
    Neq,
    Conj,
    Disj,
    Bicond,
    Imply,
    Xor,
    AE
  >;

  VFormula formula;

  template <size_t I, class FormulaType, class A2, class Alloc2>
  CUDA static VFormula create_one(const Formula<A2, Alloc2>& other, const allocator_type& allocator) {
    return VFormula::template create<I>(FormulaType(battery::get<I>(other.formula), allocator));
  }

  template <class A2, class Alloc2>
  CUDA NI static VFormula create(const Formula<A2, Alloc2>& other, const allocator_type& allocator) {
    switch(other.formula.index()) {
      case IPVarLit: return create_one<IPVarLit, PVarLit>(other, allocator);
      case INVarLit: return create_one<INVarLit, NVarLit>(other, allocator);
      case ITrue: return create_one<ITrue, True<A>>(other, allocator);
      case IFalse: return create_one<IFalse, False<A>>(other, allocator);
      case ILeq: return create_one<ILeq, Leq>(other, allocator);
      case IGt: return create_one<IGt, Gt>(other, allocator);
      case IEq: return create_one<IEq, Eq>(other, allocator);
      case INeq: return create_one<INeq, Neq>(other, allocator);
      case IConj: return create_one<IConj, Conj>(other, allocator);
      case IDisj: return create_one<IDisj, Disj>(other, allocator);
      case IBicond: return create_one<IBicond, Bicond>(other, allocator);
      case IImply: return create_one<IImply, Imply>(other, allocator);
      case IXor: return create_one<IXor, Xor>(other, allocator);
      case IAE: return create_one<IAE, AE>(other, allocator);
      default:
        printf("BUG: formula not handled.\n");
        assert(false);
        return VFormula::template create<IFalse>(False<A>());
    }
  }

  CUDA Formula(VFormula&& formula): formula(std::move(formula)) {}

  template <class F>
  CUDA NI auto forward(F&& f) const {
    switch(formula.index()) {
      case IPVarLit: return f(battery::get<IPVarLit>(formula));
      case INVarLit: return f(battery::get<INVarLit>(formula));
      case ITrue: return f(battery::get<ITrue>(formula));
      case IFalse: return f(battery::get<IFalse>(formula));
      case ILeq: return f(battery::get<ILeq>(formula));
      case IGt: return f(battery::get<IGt>(formula));
      case IEq: return f(battery::get<IEq>(formula));
      case INeq: return f(battery::get<INeq>(formula));
      case IConj: return f(battery::get<IConj>(formula));
      case IDisj: return f(battery::get<IDisj>(formula));
      case IBicond: return f(battery::get<IBicond>(formula));
      case IImply: return f(battery::get<IImply>(formula));
      case IXor: return f(battery::get<IXor>(formula));
      case IAE: return f(battery::get<IAE>(formula));
      default:
        printf("BUG: formula not handled.\n");
        assert(false);
        return f(False<A>());
    }
  }

  template <class F>
  CUDA NI auto forward(F&& f) {
    switch(formula.index()) {
      case IPVarLit: return f(battery::get<IPVarLit>(formula));
      case INVarLit: return f(battery::get<INVarLit>(formula));
      case ITrue: return f(battery::get<ITrue>(formula));
      case IFalse: return f(battery::get<IFalse>(formula));
      case ILeq: return f(battery::get<ILeq>(formula));
      case IGt: return f(battery::get<IGt>(formula));
      case IEq: return f(battery::get<IEq>(formula));
      case INeq: return f(battery::get<INeq>(formula));
      case IConj: return f(battery::get<IConj>(formula));
      case IDisj: return f(battery::get<IDisj>(formula));
      case IBicond: return f(battery::get<IBicond>(formula));
      case IImply: return f(battery::get<IImply>(formula));
      case IXor: return f(battery::get<IXor>(formula));
      case IAE: return f(battery::get<IAE>(formula));
      default:
        printf("BUG: formula not handled.\n");
        assert(false);
        False<A> false_{};
        return f(false_);
    }
  }

public:
  Formula() = default;
  Formula(this_type&&) = default;
  this_type& operator=(this_type&&) = default;

  template <class A2, class Alloc2>
  CUDA Formula(const Formula<A2, Alloc2>& other, const Allocator& allocator = Allocator())
    : formula(create(other, allocator))
  {}

  CUDA bool is(size_t kind) const {
    return formula.index() == kind;
  }

  CUDA size_t kind() const {
    return formula.index();
  }

  template <size_t I, class SubFormula>
  CUDA static this_type make(SubFormula&& sub_formula) {
    return Formula(VFormula::template create<I>(std::move(sub_formula)));
  }

  CUDA static this_type make_pvarlit(const AVar& avar) {
    return make<IPVarLit>(PVarLit(avar));
  }

  CUDA static this_type make_nvarlit(const AVar& avar) {
    return make<INVarLit>(NVarLit(avar));
  }

  CUDA static this_type make_true() {
    return make<ITrue>(True<A>{});
  }

  CUDA static this_type make_false() {
    return make<IFalse>(False<A>{});
  }

  CUDA static this_type make_leq(term_type&& left, term_type&& right) {
    return make<ILeq>(Leq(std::move(left), std::move(right)));
  }

  CUDA static this_type make_gt(term_type&& left, term_type&& right) {
    return make<IGt>(Gt(std::move(left), std::move(right)));
  }

  CUDA static this_type make_eq(term_type&& left, term_type&& right) {
    return make<IEq>(Eq(std::move(left), std::move(right)));
  }

  CUDA static this_type make_neq(term_type&& left, term_type&& right) {
    return make<INeq>(Neq(std::move(left), std::move(right)));
  }

  CUDA static this_type make_conj(this_ptr&& left, this_ptr&& right) {
    return make<IConj>(Conj(std::move(left), std::move(right)));
  }

  CUDA static this_type make_disj(this_ptr&& left, this_ptr&& right) {
    return make<IDisj>(Disj(std::move(left), std::move(right)));
  }

  CUDA static this_type make_bicond(this_ptr&& left, this_ptr&& right) {
    return make<IBicond>(Bicond(std::move(left), std::move(right)));
  }

  CUDA static this_type make_imply(this_ptr&& left, this_ptr&& right) {
    return make<IImply>(Imply(std::move(left), std::move(right)));
  }

  CUDA static this_type make_xor(this_ptr&& left, this_ptr&& right) {
    return make<IXor>(Xor(std::move(left), std::move(right)));
  }

  using tell_type = typename A::template tell_type<allocator_type>;
  using ask_type = typename A::template ask_type<allocator_type>;

  CUDA static this_type make_abstract_element(
    tell_type&& tellv, tell_type&& not_tellv,
    ask_type&& askv, ask_type&& not_askv)
  {
    return make<IAE>(AE(std::move(tellv), std::move(not_tellv), std::move(askv), std::move(not_askv)));
  }

  /** Call `deduce` iff \f$ u \leq  [\![x = 1]\!]_U \f$ and `contradeduce` iff \f$ u \leq  [\![x = 0]\!] \f$. */
  CUDA bool embed(A& a, const U& u) const {
    if(u <= U::eq_one()) { return deduce(a); }
    else if(u <= U::eq_zero()) { return contradeduce(a); }
    return false;
  }

  /** Maps the truth value of \f$ \varphi \f$ to the Boolean sublattice of `U` (see above). */
  CUDA void project(const A& a, U& r) const {
    if(a.is_bot()) { r.meet_bot(); }
    if(ask(a)) { r.meet(U::eq_one()); }
    if(nask(a)) { r.meet(U::eq_zero()); }
    r.meet(fjoin(U::eq_zero(), U::eq_one()));
  }

  CUDA void print(const A& a) const {
    forward([&](const auto& t) { t.print(a); });
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    return forward([&](const auto& t) { return t.deinterpret(a, env, apc, allocator); });
  }

  /** Given a formula \f$ \varphi \f$, the ask operation \f$ a \vDash \varphi \f$ holds whenever we can deduce \f$ \varphi \f$ from \f$ a \f$.
      More precisely, if \f$ \gamma(a) \subseteq [\![\varphi]\!]^\flat \f$, which implies that \f$ \varphi \f$ cannot remove further deduce \f$ a \f$ since \f$ a \f$ is already more precise than \f$ \varphi \f$. */
  CUDA local::B ask(const A& a) const {
    return forward([&](const auto& t) { return t.ask(a); });
  }

  /** Similar to `ask` but for \f$ \lnot{\varphi} \f$. */
  CUDA local::B nask(const A& a) const {
    return forward([&](const auto& t) { return t.nask(a); });
  }

  /** Refine the formula by supposing it must be true. */
  CUDA bool deduce(A& a) const {
    return forward([&](const auto& t) { return t.deduce(a); });
  }

  /** Refine the negation of the formula, hence we suppose the original formula needs to be false. */
  CUDA bool contradeduce(A& a) const {
    return forward([&](const auto& t) { return t.contradeduce(a); });
  }

  CUDA size_t length() const {
    return forward([](const auto& t) { return t.length(); });
  }
};

} // namespace pc
} // namespace lala

#endif