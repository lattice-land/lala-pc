// Copyright 2021 Pierre Talbot

#ifndef FORMULA_HPP
#define FORMULA_HPP

#include "arithmetic.hpp"
#include "ptr_utility.hpp"

namespace lala {

/**
 * A formula can occur in a term, e.g., `(x = 2) + (y = 2) + (z = 2) >= 2`
 * In that case, the entailment of the formula is mapped onto a sublattice of `U` supporting initialization from `0` and `1`.
 * A term can occur in a formula, as it is usually done.
 * Therefore, a formula is both a formula and a term.
 * */
template <class AD>
class Formula: public Term<AD> {
public:
  using A = AD;
  using U = typename A::Universe;
  CUDA virtual ~Formula() {}

  /** Given a formula \f$ \varphi \f$, the ask operation \f$ a \vDash \varphi \f$ holds whenever we can deduce \f$ \varphi \f$ from \f$ a \f$.
      More precisely, if \f$ \gamma(a) \subseteq [\![\varphi]\!]^\flat \f$, which implies that \f$ \varphi \f$ cannot remove further refine \f$ a \f$ since \f$ a \f$ is already more precised than \f$ \varphi \f$. */
  CUDA virtual BInc ask(const A&) const = 0;

  /** Similar to `ask` but for \f$ \lnot{\varphi} \f$. */
  CUDA virtual BInc nask(const A&) const = 0;

  /** An optional preprocessing operator that is only called at the root node.
   * This is because refinement operators are supposed to be stateless, hence are never backtracked.
   * It prepares the data of the formula to improve the refinement of subsequent calls to `refine`. */
  CUDA virtual void preprocess(A&, BInc&) = 0;

  /** Refine the formula by supposing it must be true. */
  CUDA virtual void refine(A&, BInc&) const = 0;

  /** Refine the negation of the formula, hence we suppose the original formula needs to be false. */
  CUDA virtual void nrefine(A&, BInc&) const = 0;

  CUDA virtual void tell(A&, const U&, BInc&) const = 0;
};

/** `DynFormula` wraps a term and inherits from Formula.
     A vtable will be created. */
template<class BaseFormula>
class DynFormula: public Formula<typename BaseFormula::A> {
  BaseFormula f;
public:
  using A = typename BaseFormula::A;
  using U = typename A::Universe;

  CUDA DynFormula(BaseFormula&& f): f(std::move(f)) {}
  CUDA DynFormula(DynFormula<BaseFormula>&& other): DynFormula(std::move(other.f)) {}

  CUDA BInc ask(const A& a) const {
    return f.ask(a);
  }

  CUDA BInc nask(const A& a) const {
    return f.nask(a);
  }

  CUDA void preprocess(A& a, BInc& has_changed) {
    f.preprocess(a, has_changed);
  }

  CUDA void refine(A& a, BInc& has_changed) const {
    f.refine(a, has_changed);
  }

  CUDA void nrefine(A& a, BInc& has_changed) const {
    f.nrefine(a, has_changed);
  }

  CUDA void print(const A& a) const {
    f.print(a);
  }

  CUDA void tell(A& a, const U& u, BInc& has_changed) const {
    f.tell(a, u, has_changed);
  }

  CUDA U project(const A& a) const {
    return f.project(a);
  }

  CUDA BInc is_top(const A& a) const {
    return f.is_top(a);
  }

  CUDA ~DynFormula() {}
};

/** Turn a logical formula into a term by mapping their satisfiability to and from a sublattice of `U` representing Boolean values:
 *    - Consists of the values \f$\{[\![x = 0]\!]_U \sqcap [\![x = 1]\!]_U, [\![x = 0]\!]_U, [\![x = 1]\!]_U, \top_U \}\f$.
 *    - With \f$\{[\![x = 0]\!]_U \sqcap [\![x = 1]\!]_U \f$ meaning neither true or false yet (e.g., unknown), \f$\{[\![x = 0]\!]_U \f$ modelling falsity, \f$ [\![x = 1]\!]_U \f$ modelling truth, and \f$ \top_U \f$ a logical statement both true and false (i.e., one of the variable is top). */
template<class CRTP, class F,
  class A = typename remove_ptr<F>::type::A>
struct FormulaAsTermAdapter {
  using U = typename A::Universe;

  /** Call `refine` iff \f$ u \geq  [\![x = 1]\!]_U \f$ and `nrefine` iff \f$ u \geq  [\![x = 0]\!] \f$. */
  CUDA void tell(A& a, const U& u, BInc& has_changed) const {
    if(geq<U>(u, U(1).value()).guard()) { static_cast<const CRTP*>(this)->refine(a, has_changed); }
    else if(geq<U>(u, U(0).value()).guard()) { static_cast<const CRTP*>(this)->nrefine(a, has_changed); }
  }

  /** Maps the truth value of \f$ \varphi \f$ to the Boolean sublattice of `U` (see above). */
  CUDA U project(const A& a) const {
    if(static_cast<const CRTP*>(this)->is_top(a).guard()) { return U::top(); }
    if(static_cast<const CRTP*>(this)->ask(a).guard()) { return U(1); }
    if(static_cast<const CRTP*>(this)->nask(a).guard()) { return U(0); }
    return meet(U(0), U(1));
  }
};

/** A Boolean variable defined on a universe of discourse supporting `0` for false and `1` for true. */
template<class AD, bool neg>
class VariableLiteral:
  public FormulaAsTermAdapter<VariableLiteral<AD, neg>, void, AD> {
public:
  using A = AD;
  using U = typename A::Universe;

private:
  AVar avar;

public:
  CUDA VariableLiteral(AVar avar): avar(avar) {}

private:
  template<bool negate>
  CUDA BInc ask_impl(const A& a) const {
    if constexpr(negate) {
      return geq<U>(a.project(avar), U(0).value());
    }
    else {
      return lnot(leq<U>(a.project(avar), U(0).value()));
    }
  }

public:
  /** Given a variable `x` taking a value in a universe `U` denoted by \f$ a(x) \f$.
   *   - \f$ a \vDash x \f$ holds iff \f$ \lnot (a(x) \leq [\![x = 0]\!]_U) \f$.
   *   - \f$ a \vDash \lnot{x} \f$ holds iff \f$ a(x) \geq [\![x = 0]\!]_U \f$. */
  CUDA BInc ask(const A& a) const {
    return ask_impl<neg>(a);
  }

  /** Similar to `ask` where \f$ \lnot{\lnot{x}} = x \f$. */
  CUDA BInc nask(const A& a) const {
    return ask_impl<!neg>(a);
  }

  CUDA void preprocess(A&, BInc&) {}

  /** Perform \f$ x = a(x) \sqcup [\![x = 1]\!]_U \f$. */
  CUDA void refine(A& a, BInc& has_changed) const {
    a.tell(avar, U(1), has_changed);
  }

  /** Perform \f$ x = a(x) \sqcup [\![x = 0]\!]_U \f$. */
  CUDA void nrefine(A& a, BInc& has_changed) const {
    a.tell(avar, U(0), has_changed);
  }

  CUDA BInc is_top(const A& a) const { return a.project(avar).is_top(); }

  CUDA void print(const A& a) const {
    if constexpr(neg) { printf("not "); }
    a.environment().to_lvar(avar).print();
  }
};

template<class T, class NotF = void>
class LatticeOrderPredicate;

template<class T>
class LatticeOrderPredicate<T, void> {
public:
  using T_ = typename remove_ptr<T>::type;
  using A = typename T_::A;
  using U = typename A::Universe;
  using this_type = LatticeOrderPredicate<T, void>;

protected:
  T left;
  U right;

  CUDA INLINE const T_& t() const {
    return deref(left);
  }

public:
  CUDA LatticeOrderPredicate(T&& left, U&& right): left(std::move(left)), right(std::move(right)) {}
  CUDA LatticeOrderPredicate(this_type&& other): LatticeOrderPredicate(std::move(other.left), std::move(other.right)) {}

  CUDA BInc ask(const A& a) const {
    return geq<U>(t().project(a), right.value());
  }

  CUDA BInc nask(const A&) const { assert(false); }
  CUDA void nrefine(A&, BInc&) const { assert(false); }
  CUDA void tell(A&, const U&, BInc&) const { assert(false); }
  CUDA U project(const A&) const { assert(false); return U::top(); }

  CUDA BInc is_top(const A& a) const {
    return t().is_top(a);
  }

  CUDA void preprocess(A& a, BInc& has_changed) {
    right.tell(t().project(a), has_changed);
  }

  CUDA void refine(A& a, BInc& has_changed) const {
    t().tell(a, right, has_changed);
  }

  CUDA void print(const A& a) const {
    t().print(a);
    printf(" >= ");
    right.print();
  }
};

/** A predicate of the form `t >= u` where `t` is a term, `u` an element of an abstract universe, and `>=` the lattice order of this abstract universe.
   `T` is expected to be a term (see `Term` in `terms.hpp`).

   If `NotF` is left to `void`, then project/tell/nask and nrefine of `LatticeOrderPredicate` are not available. */
template<class T, class NotF>
class LatticeOrderPredicate :
  public LatticeOrderPredicate<T, void>,
  public FormulaAsTermAdapter<LatticeOrderPredicate<T, NotF>, T>
{
public:
  using T_ = typename remove_ptr<T>::type;
  using A = typename T_::A;
  using U = typename A::Universe;
  using this_type = LatticeOrderPredicate<T, NotF>;

private:
  using base_type = LatticeOrderPredicate<T, void>;
  NotF not_f;

public:
  CUDA LatticeOrderPredicate(T&& left, U&& right, NotF&& not_f)
   : base_type(std::move(left), std::move(right)), not_f(std::move(not_f)) {}

  CUDA LatticeOrderPredicate(this_type&& other)
   : LatticeOrderPredicate(std::move(other.left), std::move(other.right), std::move(other.not_f)) {}

  CUDA BInc nask(const A& a) const {
    return deref(not_f).ask(a);
  }

  CUDA void nrefine(A& a, BInc& has_changed) const {
    deref(not_f).refine(a, has_changed);
  }

  CUDA void preprocess(A& a, BInc& has_changed) {
    base_type::preprocess(a, has_changed);
    deref(not_f).preprocess(a, has_changed);
  }

  using term_adapter = FormulaAsTermAdapter<LatticeOrderPredicate<T, NotF>, T>;
  CUDA void tell(A& a, const U& u, BInc& has_changed) const { term_adapter::tell(a, u, has_changed); }
  CUDA U project(const A& a) const { return term_adapter::project(a); }
};

template<class F, class G = F>
class Conjunction : public FormulaAsTermAdapter<Conjunction<F, G>, F> {
public:
  using F_ = typename remove_ptr<F>::type;
  using G_ = typename remove_ptr<G>::type;
  using A = typename F_::A;
  using U = typename A::Universe;
  using this_type = Conjunction<F, G>;

private:
  F f_;
  G g_;

  CUDA INLINE const F_& f() const {
    return deref(f_);
  }

  CUDA INLINE const G_& g() const {
    return deref(g_);
  }

public:
  CUDA Conjunction(F&& f, G&& g): f_(std::move(f)), g_(std::move(g)) {}
  CUDA Conjunction(this_type&& other): Conjunction(std::move(other.f_), std::move(other.g_)) {}

  CUDA BInc ask(const A& a) const {
    return land(
      f().ask(a),
      g().ask(a)
    );
  }

  CUDA BInc nask(const A& a) const {
    return lor(
      f().nask(a),
      g().nask(a)
    );
  }

  CUDA void preprocess(A& a, BInc& has_changed) {
    deref(f_).preprocess(a, has_changed);
    deref(g_).preprocess(a, has_changed);
  }

  CUDA void refine(A& a, BInc& has_changed) const {
    f().refine(a, has_changed);
    g().refine(a, has_changed);
  }

  CUDA void nrefine(A& a, BInc& has_changed) const {
    if(f().ask(a).guard()) { g().nrefine(a, has_changed); }
    else if(g().ask(a).guard()) { f().nrefine(a, has_changed); }
  }

  CUDA BInc is_top(const A& a) const {
    return lor(f().is_top(a), g().is_top(a));
  }

  CUDA void print(const A& a) const {
    f().print(a);
    printf(" /\\ ");
    g().print(a);
  }
};

template<class F, class G = F>
class Disjunction : public FormulaAsTermAdapter<Disjunction<F, G>, F> {
public:
  using F_ = typename remove_ptr<F>::type;
  using G_ = typename remove_ptr<G>::type;
  using A = typename F_::A;
  using U = typename A::Universe;
  using this_type = Disjunction<F, G>;

private:
  F f_;
  G g_;

  CUDA INLINE const F_& f() const { return deref(f_); }
  CUDA INLINE const G_& g() const { return deref(g_); }
public:
  CUDA Disjunction(F&& f, G&& g): f_(std::move(f)), g_(std::move(g)) {}
  CUDA Disjunction(this_type&& other): Disjunction(
    std::move(other.f_), std::move(other.g_)) {}

  CUDA BInc ask(const A& a) const {
    return lor(
      f().ask(a),
      g().ask(a)
    );
  }

  CUDA BInc nask(const A& a) const {
    return land(
      f().nask(a),
      g().nask(a)
    );
  }

  CUDA void preprocess(A& a, BInc& has_changed) {
    deref(f_).preprocess(a, has_changed);
    deref(g_).preprocess(a, has_changed);
  }

  CUDA void refine(A& a, BInc& has_changed) const {
    if(f().nask(a).guard()) { g().refine(a, has_changed); }
    else if(g().nask(a).guard()) { f().refine(a, has_changed); }
  }

  CUDA void nrefine(A& a, BInc& has_changed) const {
    f().nrefine(a, has_changed);
    g().nrefine(a, has_changed);
  }

  CUDA BInc is_top(const A& a) const {
    return lor(f().is_top(a), g().is_top(a));
  }

  CUDA void print(const A& a) const {
    f().print(a);
    printf(" \\/ ");
    g().print(a);
  }
};

template<class F, class G = F>
class Biconditional : public FormulaAsTermAdapter<Biconditional<F, G>, F> {
public:
  using F_ = typename remove_ptr<F>::type;
  using G_ = typename remove_ptr<G>::type;
  using A = typename F_::A;
  using U = typename A::Universe;
  using this_type = Biconditional<F, G>;

private:
  F f_;
  G g_;

  CUDA INLINE const F_& f() const { return deref(f_); }
  CUDA INLINE const G_& g() const { return deref(g_); }
public:
  CUDA Biconditional(F&& f, G&& g): f_(std::move(f)), g_(std::move(g)) {}
  CUDA Biconditional(this_type&& other): Biconditional(
    std::move(other.f_), std::move(other.g_)) {}

  CUDA BInc ask(const A& a) const {
    return lor(
      land(f().ask(a), g().ask(a)),
      land(f().nask(a), g().nask(a))
    );
  }

  // note that not(f <=> g) is equivalent to (f <=> not g)
  CUDA BInc nask(const A& a) const {
    return lor(
      land(f().ask(a), g().nask(a)),
      land(f().nask(a), g().ask(a))
    );
  }

  CUDA void preprocess(A& a, BInc& has_changed) {
    deref(f_).preprocess(a, has_changed);
    deref(g_).preprocess(a, has_changed);
  }

  CUDA void refine(A& a, BInc& has_changed) const {
    if(f().ask(a).guard()) { g().refine(a, has_changed); }
    else if(f().nask(a).guard()) { g().nrefine(a, has_changed); }
    else if(g().ask(a).guard()) { f().refine(a, has_changed); }
    else if(g().nask(a).guard()) { f().nrefine(a, has_changed); }
  }

  // note that not(f <=> g) is equivalent to (f <=> not g)
  CUDA void nrefine(A& a, BInc& has_changed) const {
    if(f().ask(a).guard()) { g().nrefine(a, has_changed); }
    else if(f().nask(a).guard()) { g().refine(a, has_changed); }
    else if(g().ask(a).guard()) { f().nrefine(a, has_changed); }
    else if(g().nask(a).guard()) { f().refine(a, has_changed); }
  }

  CUDA BInc is_top(const A& a) const {
    return lor(f().is_top(a), g().is_top(a));
  }

  CUDA void print(const A& a) const {
    f().print(a);
    printf(" <=> ");
    g().print(a);
  }
};

} // namespace lala

#endif