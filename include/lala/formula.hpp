// Copyright 2021 Pierre Talbot

#ifndef FORMULA_HPP
#define FORMULA_HPP

#include "lala/universes/primitive_upset.hpp"
#include "ptr_utility.hpp"

namespace lala {
namespace pc {
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
  using U = typename A::local_universe;
  CUDA virtual ~Formula() {}

  /** Given a formula \f$ \varphi \f$, the ask operation \f$ a \vDash \varphi \f$ holds whenever we can deduce \f$ \varphi \f$ from \f$ a \f$.
      More precisely, if \f$ \gamma(a) \subseteq [\![\varphi]\!]^\flat \f$, which implies that \f$ \varphi \f$ cannot remove further refine \f$ a \f$ since \f$ a \f$ is already more precise than \f$ \varphi \f$. */
  CUDA virtual local::BInc ask(const A&) const = 0;

  /** Similar to `ask` but for \f$ \lnot{\varphi} \f$. */
  CUDA virtual local::BInc nask(const A&) const = 0;

  /** An optional preprocessing operator that is only called at the root node.
   * This is because refinement operators are supposed to be stateless, hence are never backtracked.
   * It prepares the data of the formula to improve the refinement of subsequent calls to `refine`. */
  CUDA virtual void preprocess(A&, local::BInc&) = 0;

  /** Refine the formula by supposing it must be true. */
  CUDA virtual void refine(A&, local::BInc&) const = 0;

  /** Refine the negation of the formula, hence we suppose the original formula needs to be false. */
  CUDA virtual void nrefine(A&, local::BInc&) const = 0;

  CUDA virtual void tell(A&, const U&, local::BInc&) const = 0;
};

/** `DynFormula` wraps a term and inherits from Formula.
     A vtable will be created. */
template<class BaseFormula>
class DynFormula: public Formula<typename BaseFormula::A> {
  BaseFormula f;
public:
  using A = typename BaseFormula::A;
  using U = typename A::local_universe;

  CUDA DynFormula(BaseFormula&& f): f(std::move(f)) {}
  CUDA DynFormula(DynFormula<BaseFormula>&& other): DynFormula(std::move(other.f)) {}

  CUDA local::BInc ask(const A& a) const {
    return f.ask(a);
  }

  CUDA local::BInc nask(const A& a) const {
    return f.nask(a);
  }

  CUDA void preprocess(A& a, local::BInc& has_changed) {
    f.preprocess(a, has_changed);
  }

  CUDA void refine(A& a, local::BInc& has_changed) const {
    f.refine(a, has_changed);
  }

  CUDA void nrefine(A& a, local::BInc& has_changed) const {
    f.nrefine(a, has_changed);
  }

  CUDA void print(const A& a) const {
    f.print(a);
  }

  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    f.tell(a, u, has_changed);
  }

  CUDA U project(const A& a) const {
    return f.project(a);
  }

  CUDA local::BInc is_top(const A& a) const {
    return f.is_top(a);
  }

  CUDA TFormula<battery::standard_allocator> deinterpret() const {
    return f.deinterpret();
  }

  CUDA ~DynFormula() {}
};

/** Turn a logical formula into a term by mapping their satisfiability to and from a sublattice of `U` representing Boolean values:
 *    - Consists of four distinct values \f$\{[\![x = 0]\!]_U \sqcap [\![x = 1]\!]_U, [\![x = 0]\!]_U, [\![x = 1]\!]_U, \top_U \}\f$.
 *    - With \f$\{[\![x = 0]\!]_U \sqcap [\![x = 1]\!]_U \f$ meaning neither true or false yet (e.g., unknown), \f$\{[\![x = 0]\!]_U \f$ modelling falsity, \f$ [\![x = 1]\!]_U \f$ modelling truth, and \f$ \top_U \f$ a logical statement both true and false (i.e., one of the variable is top). */
template<class CRTP, class F,
  class A = typename remove_ptr<F>::type::A>
struct FormulaAsTermAdapter {
  using U = typename A::local_universe;

  /** Call `refine` iff \f$ u \geq  [\![x = 1]\!]_U \f$ and `nrefine` iff \f$ u \geq  [\![x = 0]\!] \f$. */
  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    if(u >= U::eq_one()) { static_cast<const CRTP*>(this)->refine(a, has_changed); }
    else if(u >= U::eq_zero()) { static_cast<const CRTP*>(this)->nrefine(a, has_changed); }
  }

  /** Maps the truth value of \f$ \varphi \f$ to the Boolean sublattice of `U` (see above). */
  CUDA U project(const A& a) const {
    if(static_cast<const CRTP*>(this)->is_top(a)) { return U::top(); }
    if(static_cast<const CRTP*>(this)->ask(a)) { return U::eq_one(); }
    if(static_cast<const CRTP*>(this)->nask(a)) { return U::eq_zero(); }
    constexpr auto unknown = meet(U::eq_zero(), U::eq_one());
    return unknown;
  }
};

/** A Boolean variable defined on a universe of discourse supporting `0` for false and `1` for true. */
template<class AD, bool neg>
class VariableLiteral:
  public FormulaAsTermAdapter<VariableLiteral<AD, neg>, void, AD> {
public:
  using A = AD;
  using U = typename A::local_universe;

private:
  AVar avar;

public:
  CUDA VariableLiteral(AVar avar): avar(avar) {}

private:
  template<bool negate>
  CUDA local::BInc ask_impl(const A& a) const {
    if constexpr(negate) {
      return a.project(avar) >= U::eq_zero();
    }
    else {
      return !(a.project(avar) <= U::eq_zero());
    }
  }

public:
  /** Given a variable `x` taking a value in a universe `U` denoted by \f$ a(x) \f$.
   *   - \f$ a \vDash x \f$ holds iff \f$ \lnot (a(x) \leq [\![x = 0]\!]_U) \f$.
   *   - \f$ a \vDash \lnot{x} \f$ holds iff \f$ a(x) \geq [\![x = 0]\!]_U \f$. */
  CUDA local::BInc ask(const A& a) const {
    return ask_impl<neg>(a);
  }

  /** Similar to `ask` where \f$ \lnot{\lnot{x}} = x \f$. */
  CUDA local::BInc nask(const A& a) const {
    return ask_impl<!neg>(a);
  }

  CUDA void preprocess(A&, local::BInc&) {}

  /** Perform \f$ x = a(x) \sqcup [\![x = 1]\!]_U \f$. */
  CUDA void refine(A& a, local::BInc& has_changed) const {
    a.tell(avar, U::eq_one(), has_changed);
  }

  /** Perform \f$ x = a(x) \sqcup [\![x = 0]\!]_U \f$. */
  CUDA void nrefine(A& a, local::BInc& has_changed) const {
    a.tell(avar, U::eq_zero(), has_changed);
  }

  CUDA local::BInc is_top(const A& a) const { return a.project(avar).is_top(); }

  CUDA void print(const A& a) const {
    if constexpr(neg) { printf("not "); }
    printf("(%d,%d)", avar.aty(), avar.vid());
  }

  CUDA TFormula<battery::standard_allocator> deinterpret() const {
    auto f = TFormula<battery::standard_allocator>::make_avar(avar);
    f.type_as(UNTYPED); // The variable should be interpreted as an atomic formula in PC (not in VStore).
    return std::move(f);
  }
};

template<class T, class NotF = void>
class LatticeOrderPredicate;

template<class T>
class LatticeOrderPredicate<T, void> {
public:
  using T_ = typename remove_ptr<T>::type;
  using A = typename T_::A;
  using U = typename A::local_universe;
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

  CUDA local::BInc ask(const A& a) const {
    return t().project(a) >= right;
  }

  CUDA local::BInc nask(const A&) const { assert(false); return local::BInc::bot(); }
  CUDA void nrefine(A&, local::BInc&) const { assert(false); }
  CUDA void tell(A&, const U&, local::BInc&) const { assert(false); }
  CUDA U project(const A&) const { assert(false); return U::top(); }

  CUDA local::BInc is_top(const A& a) const {
    return t().is_top(a);
  }

  CUDA void preprocess(A& a, local::BInc& has_changed) {
    right.tell(t().project(a), has_changed);
  }

  CUDA void refine(A& a, local::BInc& has_changed) const {
    t().tell(a, right, has_changed);
  }

  CUDA void print(const A& a) const {
    t().print(a);
    printf(" >= ");
    right.print();
  }
private:
  using F = TFormula<battery::standard_allocator>;
  using Seq = typename F::Sequence;

  CUDA Seq map_seq(const Seq& seq, const F& t) const {
    Seq seq2;
    for(int i = 0; i < seq.size(); ++i) {
      seq2.push_back(map_avar(seq[i], t));
    }
    return std::move(seq2);
  }

  CUDA F map_avar(const F& f, const F& t) const {
    switch(f.index()) {
      case F::V: return t;
      case F::Seq: return F::make_nary(f.sig(), map_seq(f.seq(), t), f.type());
      case F::ESeq: return F::make_nary(f.esig(), map_seq(f.eseq(), t), f.type());
      default: return f;
    }
  }

public:
  CUDA TFormula<battery::standard_allocator> deinterpret() const {
    // We deinterpret the constant with a placeholder variable that is then replaced by the interpretation of the left term.
    VarEnv<battery::standard_allocator> empty_env;
    auto uf = right.deinterpret(AVar{}, empty_env);
    auto tf = t().deinterpret();
    return map_avar(uf, tf);
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
  using U = typename A::local_universe;
  using this_type = LatticeOrderPredicate<T, NotF>;

private:
  using base_type = LatticeOrderPredicate<T, void>;
  NotF not_f;

public:
  CUDA LatticeOrderPredicate(T&& left, U&& right, NotF&& not_f)
   : base_type(std::move(left), std::move(right)), not_f(std::move(not_f)) {}

  CUDA LatticeOrderPredicate(this_type&& other)
   : LatticeOrderPredicate(std::move(other.left), std::move(other.right), std::move(other.not_f)) {}

  CUDA local::BInc nask(const A& a) const {
    return deref(not_f).ask(a);
  }

  CUDA void nrefine(A& a, local::BInc& has_changed) const {
    deref(not_f).refine(a, has_changed);
  }

  CUDA void preprocess(A& a, local::BInc& has_changed) {
    base_type::preprocess(a, has_changed);
    deref(not_f).preprocess(a, has_changed);
  }

  using term_adapter = FormulaAsTermAdapter<LatticeOrderPredicate<T, NotF>, T>;
  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const { term_adapter::tell(a, u, has_changed); }
  CUDA U project(const A& a) const { return term_adapter::project(a); }
};

template<class F, class G = F>
class Conjunction : public FormulaAsTermAdapter<Conjunction<F, G>, F> {
public:
  using F_ = typename remove_ptr<F>::type;
  using G_ = typename remove_ptr<G>::type;
  using A = typename F_::A;
  using U = typename A::local_universe;
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

  CUDA local::BInc ask(const A& a) const {
    return f().ask(a) && g().ask(a);
  }

  CUDA local::BInc nask(const A& a) const {
    return f().nask(a) || g().nask(a);
  }

  CUDA void preprocess(A& a, local::BInc& has_changed) {
    deref(f_).preprocess(a, has_changed);
    deref(g_).preprocess(a, has_changed);
  }

  CUDA void refine(A& a, local::BInc& has_changed) const {
    f().refine(a, has_changed);
    g().refine(a, has_changed);
  }

  CUDA void nrefine(A& a, local::BInc& has_changed) const {
    if(f().ask(a)) { g().nrefine(a, has_changed); }
    else if(g().ask(a)) { f().nrefine(a, has_changed); }
  }

  CUDA local::BInc is_top(const A& a) const {
    return f().is_top(a) || g().is_top(a);
  }

  CUDA void print(const A& a) const {
    f().print(a);
    printf(" /\\ ");
    g().print(a);
  }

  CUDA TFormula<battery::standard_allocator> deinterpret() const {
    auto left = f().deinterpret();
    auto right = g().deinterpret();
    if(left.is_true()) { return right; }
    else if(right.is_true()) { return left; }
    else if(left.is_false()) { return left; }
    else if(right.is_false()) { return right; }
    else {
      return TFormula<battery::standard_allocator>::make_binary(left, AND, right);
    }
  }
};

template<class F, class G = F>
class Disjunction : public FormulaAsTermAdapter<Disjunction<F, G>, F> {
public:
  using F_ = typename remove_ptr<F>::type;
  using G_ = typename remove_ptr<G>::type;
  using A = typename F_::A;
  using U = typename A::local_universe;
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

  CUDA local::BInc ask(const A& a) const {
    return f().ask(a) || g().ask(a);
  }

  CUDA local::BInc nask(const A& a) const {
    return f().nask(a) && g().nask(a);
  }

  CUDA void preprocess(A& a, local::BInc& has_changed) {
    deref(f_).preprocess(a, has_changed);
    deref(g_).preprocess(a, has_changed);
  }

  CUDA void refine(A& a, local::BInc& has_changed) const {
    if(f().nask(a)) { g().refine(a, has_changed); }
    else if(g().nask(a)) { f().refine(a, has_changed); }
  }

  CUDA void nrefine(A& a, local::BInc& has_changed) const {
    f().nrefine(a, has_changed);
    g().nrefine(a, has_changed);
  }

  CUDA local::BInc is_top(const A& a) const {
    return f().is_top(a) || g().is_top(a);
  }

  CUDA void print(const A& a) const {
    f().print(a);
    printf(" \\/ ");
    g().print(a);
  }

  CUDA TFormula<battery::standard_allocator> deinterpret() const {
    auto left = f().deinterpret();
    auto right = g().deinterpret();
    if(left.is_true()) { return left; }
    else if(right.is_true()) { return right; }
    else if(left.is_false()) { return right; }
    else if(right.is_false()) { return left; }
    else {
      return TFormula<battery::standard_allocator>::make_binary(left, OR, right);
    }
  }
};

template<class F, class G = F>
class Biconditional : public FormulaAsTermAdapter<Biconditional<F, G>, F> {
public:
  using F_ = typename remove_ptr<F>::type;
  using G_ = typename remove_ptr<G>::type;
  using A = typename F_::A;
  using U = typename A::local_universe;
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

  CUDA local::BInc ask(const A& a) const {
    return
      (f().ask(a) && g().ask(a)) ||
      (f().nask(a) && g().nask(a));
  }

  // note that not(f <=> g) is equivalent to (f <=> not g)
  CUDA local::BInc nask(const A& a) const {
    return
      (f().ask(a) && g().nask(a)) ||
      (f().nask(a) && g().ask(a));
  }

  CUDA void preprocess(A& a, local::BInc& has_changed) {
    deref(f_).preprocess(a, has_changed);
    deref(g_).preprocess(a, has_changed);
  }

  CUDA void refine(A& a, local::BInc& has_changed) const {
    if(f().ask(a)) { g().refine(a, has_changed); }
    else if(f().nask(a)) { g().nrefine(a, has_changed); }
    else if(g().ask(a)) { f().refine(a, has_changed); }
    else if(g().nask(a)) { f().nrefine(a, has_changed); }
  }

  // note that not(f <=> g) is equivalent to (f <=> not g)
  CUDA void nrefine(A& a, local::BInc& has_changed) const {
    if(f().ask(a)) { g().nrefine(a, has_changed); }
    else if(f().nask(a)) { g().refine(a, has_changed); }
    else if(g().ask(a)) { f().nrefine(a, has_changed); }
    else if(g().nask(a)) { f().refine(a, has_changed); }
  }

  CUDA local::BInc is_top(const A& a) const {
    return f().is_top(a) || g().is_top(a);
  }

  CUDA void print(const A& a) const {
    f().print(a);
    printf(" <=> ");
    g().print(a);
  }

  CUDA TFormula<battery::standard_allocator> deinterpret() const {
    return TFormula<battery::standard_allocator>::make_binary(f().deinterpret(), EQUIV, g().deinterpret());
  }
};

} // namespace pc
} // namespace lala

#endif