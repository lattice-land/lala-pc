// Copyright 2021 Pierre Talbot

#ifndef LALA_PC_TERMS_HPP
#define LALA_PC_TERMS_HPP

#include "battery/vector.hpp"
#include "lala/universes/arith_bound.hpp"

namespace lala {
namespace pc {

template <class AD, class Allocator>
class Term;

template <class AD, class Allocator>
class Formula;

template <class AD>
class Constant {
public:
  using A = AD;
  using U = typename A::local_universe;

private:
  U k;

public:
  CUDA Constant(U&& k) : k(k) {}
  Constant(Constant<A>&& other) = default;

  template <class A2, class Alloc>
  CUDA Constant(const Constant<A2>& other, const Alloc&): k(other.k) {}

  CUDA bool embed(A&, const U&) const { return false; }
  CUDA void project(const A&, U& r) const { r.meet(k); }
  CUDA void print(const A&) const { ::battery::print(k); }
  template <class Env, class Allocator = typename Env::allocator_type>
  CUDA TFormula<Allocator> deinterpret(const A&, const Env&, AType, Allocator allocator = Allocator()) const {
    return k.template deinterpret<TFormula<Allocator>>(allocator);
  }
  CUDA size_t length() const { return 1; }

  template <class A2>
  friend class Constant;
};

template <class AD>
class Variable {
public:
  using A = AD;
  using U = typename A::local_universe;

private:
  AVar avar;

public:
  CUDA Variable() {}
  CUDA Variable(AVar avar) : avar(avar) {}
  Variable(Variable<A>&& other) = default;
  Variable<A>& operator=(Variable<A>&& other) = default;

  template <class A2, class Alloc>
  CUDA Variable(const Variable<A2>& other, const Alloc&): avar(other.avar) {}

  CUDA bool embed(A& a, const U& u) const {
    return a.embed(avar, u);
  }

  CUDA void project(const A& a, U& r) const {
    return a.project(avar, r);
  }

  CUDA void print(const A& a) const { printf("(%d,%d)", avar.aty(), avar.vid()); }

  template <class Env, class Allocator = typename Env::allocator_type>
  CUDA TFormula<Allocator> deinterpret(const A&, const Env& env, AType, Allocator allocator = Allocator()) const {
    using F = TFormula<Allocator>;
    return F::make_lvar(avar.aty(), LVar<Allocator>(env.name_of(avar), allocator));
  }

  CUDA size_t length() const { return 1; }

  template <class A2>
  friend class Variable;
};

template<class Universe>
struct NegOp {
  using U = Universe;

  CUDA static void project(const U& a, U& r) {
    r.project(NEG, a);
  }

  CUDA static void residual(const U& a, U& r) {
    project(a, r); // negation is its own residual.
  }

  static constexpr bool function_symbol = false;
  CUDA static const char* symbol() { return "-"; }
  CUDA static Sig sig() { return NEG; }
};

template<class Universe>
struct AbsOp {
  using U = Universe;

  CUDA static void project(const U& a, U& r) {
    r.project(ABS, a);
  }

  CUDA static void residual(const U& a, U& r) {
    r.meet(fjoin(a, project_fun(NEG, a)));
  }

  static constexpr bool function_symbol = true;
  CUDA static const char* symbol() { return "abs"; }
  CUDA static Sig sig() { return ABS; }
};

template <class AD, class UnaryOp, class Allocator>
class Unary {
public:
  using allocator_type = Allocator;
  using A = AD;
  using U = typename A::local_universe;
  using this_type = Unary<A, allocator_type, UnaryOp>;

  template <class A2, class UnaryOp2, class Alloc2>
  friend class Unary;

private:
  using sub_type = Term<A, allocator_type>;
  battery::unique_ptr<sub_type, allocator_type> x_term;

  CUDA INLINE const sub_type& x() const {
    return *x_term;
  }

public:
  CUDA Unary(battery::unique_ptr<sub_type, allocator_type>&& x_term): x_term(std::move(x_term)) {}
  CUDA Unary(this_type&& other): Unary(std::move(other.x_term)) {}

  template <class A2, class UnaryOp2, class Alloc2>
  CUDA Unary(const Unary<A2, UnaryOp2, Alloc2>& other, const allocator_type& allocator):
    x_term(battery::allocate_unique<sub_type>(allocator, *other.x_term))
  {}

  CUDA bool embed(A& a, const U& u) const {
    U tmp{};
    UnaryOp::residual(u, tmp);
    return x().embed(a, tmp);
  }

  CUDA void project(const A& a, U& r) const {
    U tmp{};
    x().project(a, tmp);
    UnaryOp::project(tmp, r);
  }

  CUDA NI void print(const A& a) const {
    printf("%s", UnaryOp::symbol());
    if constexpr(UnaryOp::function_symbol) { printf("("); }
    x().print(a);
    if constexpr(UnaryOp::function_symbol) { printf(")"); }
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    using F = TFormula<Allocator2>;
    return F::make_unary(UnaryOp::sig(), x().deinterpret(a, env, apc, allocator), apc, allocator);
  }

  CUDA size_t length() const { return 1 + x().length(); }
};

template<class Universe>
struct GroupAdd {
  using U = Universe;
  static constexpr bool has_absorbing_element = false;

  CUDA static void project(const U& a, const U& b, U& r) {
    r.project(ADD, a, b);
  }

  CUDA static bool is_absorbing(const U& a) {
    return false;
  }

  CUDA static void rev_op(const U& a, const U& b, U& r) {
    U tmp{};
    tmp.additive_inverse(b);
    project(a, tmp, r);
  }

  CUDA static void left_residual(const U& a, const U& b, U& r) {
    r.project(SUB, a, b);
  }

  CUDA static void right_residual(const U& a, const U& b, U& r) {
    left_residual(a, b, r);
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '+'; }
  CUDA static Sig sig() { return ADD; }
};

template<class Universe>
struct GroupSub {
  using U = Universe;
  static constexpr bool has_absorbing_element = false;

  CUDA static void project(const U& a, const U& b, U& r) {
    return r.project(SUB, a, b);
  }

  CUDA static void left_residual(const U& a, const U& b, U& r) {
    return r.project(ADD, a, b);
  }

  CUDA static void right_residual(const U& a, const U& b, U& r) {
    return r.project(SUB, b, a);
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '-'; }
  CUDA static Sig sig() { return SUB; }
};

template<class Universe, Sig divsig>
struct GroupMul {
  using U = Universe;

  CUDA static void project(const U& a, const U& b, U& r) {
    r.project(MUL, a, b);
  }

  CUDA static bool is_absorbing(const U& a) {
    return a == U::eq_zero();
  }

  /** \pre `is_absorbing(b)` must be `false`. */
  CUDA static void rev_op(const U& a, const U& b, U& r) {
    return r.project(divsig, a, b);
  }

  /** If `a` and `b` contains 0, then we cannot say anything on the inverse since 0 is absorbing and the inverse could be anything. */
  CUDA static void left_residual(const U& a, const U& b, U& r) {
    if(!(a >= U::eq_zero() && b >= U::eq_zero())) {
      r.project(divsig, a, b);
      a.print(); printf(" \\ "); b.print(); printf(" = "); r.print(); printf("\n");
    }
  }

  CUDA static void right_residual(const U& a, const U& b, U& r) {
    left_residual(a, b, r);
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '*'; }
  CUDA static Sig sig() { return MUL; }
};

template<class Universe, Sig divsig>
struct GroupDiv {
  using U = Universe;

  CUDA static void project(const U& a, const U& b, U& r) {
    return r.project(divsig, a, b);
  }

  CUDA static void left_residual(const U& a, const U& b, U& r) {
    return r.project(MUL, a, b);
  }

  CUDA static void right_residual(const U& a, const U& b, U& r) {
    if(!(b >= U::eq_zero())) {
      r.project(divsig, b, a);
    }
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '/'; }
  CUDA static Sig sig() { return divsig; }
};

template<class Universe, Sig msig>
struct GroupMinMax {
  static_assert(msig == MIN || msig == MAX);

  using U = Universe;
  CUDA static void project(const U& a, const U& b, U& r) {
    return r.project(msig, a, b);
  }

  CUDA static void left_residual(const U& a, const U& b, U& r) {
    if(fmeet(a, b).is_bot()) {
      r.meet(a);
    }
  }

  CUDA static void right_residual(const U& a, const U& b, U& r) {
    left_residual(a, b, r);
  }

  static constexpr bool prefix_symbol = true;
  CUDA static const char* symbol() { return msig == MIN ? "min" : "max"; }
  CUDA static Sig sig() { return msig; }
};

template <class AD, class Group, class Allocator>
class Binary {
public:
  using A = AD;
  using allocator_type = Allocator;
  using U = typename Group::U;
  using G = Group;
  using this_type = Binary<A, G, allocator_type>;

  template <class A2, class Group2, class Alloc2>
  friend class Binary;

  using sub_type = Term<A, allocator_type>;
  using sub_ptr = battery::unique_ptr<sub_type, allocator_type>;

private:
  sub_ptr x_term;
  sub_ptr y_term;

  CUDA INLINE const sub_type& x() const {
    return *x_term;
  }

  CUDA INLINE const sub_type& y() const {
    return *y_term;
  }

public:
  CUDA Binary(sub_ptr&& x_term, sub_ptr&& y_term)
    : x_term(std::move(x_term))
    , y_term(std::move(y_term)) {}

  CUDA Binary(this_type&& other)
    : Binary(std::move(other.x_term), std::move(other.y_term)) {}

  template <class A2, class Group2, class Alloc2>
  CUDA Binary(const Binary<A2, Group2, Alloc2>& other, const allocator_type& allocator)
    : x_term(battery::allocate_unique<sub_type>(allocator, *other.x_term))
    , y_term(battery::allocate_unique<sub_type>(allocator, *other.y_term))
  {}

  /** Enforce `x <op> y <= u` where <= is the lattice order of the underlying abstract universe.
      For instance, over the interval abstract universe, `x + y <= [2..5]` will ensure that `x + y` is eventually at least `2` and at most `5`. */
  CUDA bool embed(A& a, const U& u) const {
    U xt{};
    U yt{};
    U residual{};
    bool has_changed = false;
    printf("embed("); u.print(); printf(", "); print(a); printf(")\n");
    if(!x().is(sub_type::IConstant)) {
      y().project(a, yt);
      printf("project("); y().print(a); printf(") = "); yt.print(); printf("\n");
      G::left_residual(u, yt, residual);
      printf("left_residual("); u.print(); printf(", "); yt.print(); printf(") = "); residual.print(); printf("\n");
      has_changed |= x().embed(a, residual);   // x <- u <residual> y
      x().project(a, xt);
      printf("project("); x().print(a); printf(") = "); xt.print(); printf("\n");
    }
    if(!y().is(sub_type::IConstant)) {
      x().project(a, xt);
      printf("project("); x().print(a); printf(") = "); xt.print(); printf("\n");
      residual.join_top();
      G::right_residual(u, xt, residual);
      printf("right_residual("); u.print(); printf(", "); xt.print(); printf(") = "); residual.print(); printf("\n");
      has_changed |= y().embed(a, residual);   // y <- u <residual> x
      y().project(a, yt);
      printf("project("); y().print(a); printf(") = "); yt.print(); printf("\n");
    }
    return has_changed;
  }

  CUDA void project(const A& a, U& r) const {
    U xt{};
    U yt{};
    x().project(a, xt);
    y().project(a, yt);
    G::project(xt, yt, r);
  }

  CUDA NI void print(const A& a) const {
    if constexpr(G::prefix_symbol) {
      printf("%s(", G::symbol());
      x().print(a);
      printf(", ");
      x().print(a);
      printf(")");
    }
    else {
      x().print(a);
      printf(" %c ", G::symbol());
      y().print(a);
    }
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA NI TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    using F = TFormula<Allocator2>;
    return F::make_binary(
      x().deinterpret(a, env, apc, allocator),
      G::sig(),
      y().deinterpret(a, env, apc, allocator),
      apc,
      allocator);
  }

  CUDA size_t length() const { return 1 + x().length() + y().length(); }
};

// Nary is only valid for commutative group (e.g., addition and multiplication).
template <class Combinator>
class Nary {
public:
  using this_type = Nary<Combinator>;
  using allocator_type = typename Combinator::allocator_type;
  using A = typename Combinator::A;
  using U = typename Combinator::U;
  using G = typename Combinator::G;

  template <class Combinator2>
  friend class Nary;
private:
  using sub_type = Term<A, allocator_type>;
  battery::vector<sub_type, allocator_type> terms;

  CUDA INLINE const sub_type& t(size_t i) const {
    return terms[i];
  }

public:
  CUDA Nary(battery::vector<sub_type, allocator_type>&& terms): terms(std::move(terms)) {}
  CUDA Nary(this_type&& other): Nary(std::move(other.terms)) {}

  template <class Combinator2>
  CUDA Nary(const Nary<Combinator2>& other, const allocator_type& allocator)
    : terms(other.terms, allocator)
  {}

  CUDA void project(const A& a, U& r) const {
    U tmp{};
    U tmp2{};
    U accu{};
    t(0).project(a, accu);
    for(int i = 1; i < terms.size(); ++i) {
      t(i).project(a, tmp);
      G::project(accu, tmp, tmp2);
      accu = tmp2;
      tmp.join_top();
      tmp2.join_top();
    }
    r.meet(accu);
  }

  CUDA bool embed(A& a, const U& u) const {
    U all{};
    U tmp{};
    U tmp2{};
    U residual{};
    bool has_changed = false;
    project(a, all);
    if(!G::is_absorbing(all)) {
      for(int i = 0; i < terms.size(); ++i) {
        t(i).project(a, tmp);
        G::rev_op(all, tmp, tmp2);
        G::left_residual(u, tmp2, residual);
        has_changed |= t(i).embed(a, residual);
      }
    }
    return has_changed;
  }

  CUDA NI void print(const A& a) const {
    t(0).print(a);
    for(int i = 1; i < terms.size(); ++i) {
      printf(" %c ", G::symbol());
      t(i).print(a);
    }
  }

  template <class Env, class Allocator = typename Env::allocator_type>
  CUDA NI TFormula<Allocator> deinterpret(const A& a, const Env& env, AType apc, Allocator allocator = Allocator()) const {
    using F = TFormula<Allocator>;
    typename F::Sequence seq = typename F::Sequence(allocator);
    for(int i = 0; i < terms.size(); ++i) {
      seq.push_back(t(i).deinterpret(a, env, apc, allocator));
    }
    return F::make_nary(G::sig(), std::move(seq), apc);
  }

  CUDA size_t length() const {
    size_t len = 1;
    for(int i = 0; i < terms.size(); ++i) {
      len += t(i).length();
    }
    return len;
  }
};

template <class AD, class Allocator>
class Term {
public:
  using A = AD;
  using U = typename A::local_universe;
  using allocator_type = Allocator;
  using this_type = Term<A, allocator_type>;
  using this_ptr = battery::unique_ptr<Term<A, allocator_type>, allocator_type>;
  using formula_ptr = battery::unique_ptr<Formula<A, allocator_type>, allocator_type>;
  using Neg = Unary<A, NegOp<U>, allocator_type>;
  using Abs = Unary<A, AbsOp<U>, allocator_type>;
  using Add = Binary<A, GroupAdd<U>, allocator_type>;
  using Sub = Binary<A, GroupSub<U>, allocator_type>;
  using Mul = Binary<A, GroupMul<U, EDIV>, allocator_type>;
  using TDiv = Binary<A, GroupDiv<U, TDIV>, allocator_type>;
  using FDiv = Binary<A, GroupDiv<U, FDIV>, allocator_type>;
  using CDiv = Binary<A, GroupDiv<U, CDIV>, allocator_type>;
  using EDiv = Binary<A, GroupDiv<U, EDIV>, allocator_type>;
  using Min = Binary<A, GroupMinMax<U, MIN>, allocator_type>;
  using Max = Binary<A, GroupMinMax<U, MAX>, allocator_type>;
  using NaryAdd = Nary<Add>;
  using NaryMul = Nary<Mul>;

  static constexpr size_t IVar = 0;
  static constexpr size_t IConstant = IVar + 1;
  static constexpr size_t IFormula = IConstant + 1;
  static constexpr size_t INeg = IFormula + 1;
  static constexpr size_t IAbs = INeg + 1;
  static constexpr size_t IAdd = IAbs + 1;
  static constexpr size_t ISub = IAdd + 1;
  static constexpr size_t IMul = ISub + 1;
  static constexpr size_t ITDiv = IMul + 1;
  static constexpr size_t IFDiv = ITDiv + 1;
  static constexpr size_t ICDiv = IFDiv + 1;
  static constexpr size_t IEDiv = ICDiv + 1;
  static constexpr size_t IMin = IEDiv + 1;
  static constexpr size_t IMax = IMin + 1;
  static constexpr size_t INaryAdd = IMax + 1;
  static constexpr size_t INaryMul = INaryAdd + 1;

  template <class A2, class Alloc2>
  friend class Term;

private:
  using VTerm = battery::variant<
    Variable<A>,
    Constant<A>,
    formula_ptr,
    Neg,
    Abs,
    Add,
    Sub,
    Mul,
    TDiv,
    FDiv,
    CDiv,
    EDiv,
    Min,
    Max,
    NaryAdd,
    NaryMul
  >;

  VTerm term;

  template <size_t I, class TermType, class A2, class Alloc2>
  CUDA static VTerm create_one(const Term<A2, Alloc2>& other, const allocator_type& allocator) {
    return VTerm::template create<I>(TermType(battery::get<I>(other.term), allocator));
  }

  template <class A2, class Alloc2>
  CUDA NI static VTerm create(const Term<A2, Alloc2>& other, const allocator_type& allocator) {
    switch(other.term.index()) {
      case IVar: return create_one<IVar, Variable<A>>(other, allocator);
      case IConstant: return create_one<IConstant, Constant<A>>(other, allocator);
      case IFormula:
        return VTerm::template create<IFormula>(battery::allocate_unique<Formula<A, allocator_type>>(
          allocator, *battery::get<IFormula>(other.term)));
      case INeg: return create_one<INeg, Neg>(other, allocator);
      case IAbs: return create_one<IAbs, Abs>(other, allocator);
      case IAdd: return create_one<IAdd, Add>(other, allocator);
      case ISub: return create_one<ISub, Sub>(other, allocator);
      case IMul: return create_one<IMul, Mul>(other, allocator);
      case ITDiv: return create_one<ITDiv, TDiv>(other, allocator);
      case IFDiv: return create_one<IFDiv, FDiv>(other, allocator);
      case ICDiv: return create_one<ICDiv, CDiv>(other, allocator);
      case IEDiv: return create_one<IEDiv, EDiv>(other, allocator);
      case IMin: return create_one<IMin, Min>(other, allocator);
      case IMax: return create_one<IMax, Max>(other, allocator);
      case INaryAdd: return create_one<INaryAdd, NaryAdd>(other, allocator);
      case INaryMul: return create_one<INaryMul, NaryMul>(other, allocator);
      default:
        printf("BUG: term not handled.\n");
        assert(false);
        return VTerm::template create<IVar>(Variable<A>());
    }
  }

  CUDA Term(VTerm&& term): term(std::move(term)) {}

  template <class F>
  CUDA NI auto forward(F&& f) const {
    switch(term.index()) {
      case IVar: return f(battery::get<IVar>(term));
      case IConstant: return f(battery::get<IConstant>(term));
      case IFormula: return f(*battery::get<IFormula>(term));
      case INeg: return f(battery::get<INeg>(term));
      case IAbs: return f(battery::get<IAbs>(term));
      case IAdd: return f(battery::get<IAdd>(term));
      case ISub: return f(battery::get<ISub>(term));
      case IMul: return f(battery::get<IMul>(term));
      case ITDiv: return f(battery::get<ITDiv>(term));
      case IFDiv: return f(battery::get<IFDiv>(term));
      case ICDiv: return f(battery::get<ICDiv>(term));
      case IEDiv: return f(battery::get<IEDiv>(term));
      case IMin: return f(battery::get<IMin>(term));
      case IMax: return f(battery::get<IMax>(term));
      case INaryAdd: return f(battery::get<INaryAdd>(term));
      case INaryMul: return f(battery::get<INaryMul>(term));
      default:
        printf("BUG: term not handled.\n");
        assert(false);
        return f(Variable<A>());
    }
  }

public:
  Term() = default;
  Term(this_type&&) = default;
  this_type& operator=(this_type&&) = default;

  template <class A2, class Alloc2>
  CUDA Term(const Term<A2, Alloc2>& other, const allocator_type& allocator = allocator_type())
    : term(create(other, allocator))
  {}

  CUDA bool is(size_t kind) const {
    return term.index() == kind;
  }

  template <size_t I, class SubTerm>
  CUDA static this_type make(SubTerm&& sub_term) {
    return Term(VTerm::template create<I>(std::move(sub_term)));
  }

  CUDA static this_type make_var(const AVar& avar) {
    return make<IVar>(Variable<A>(avar));
  }

  CUDA static this_type make_constant(U&& sub_term) {
    return make<IConstant>(Constant<A>(std::move(sub_term)));
  }

  CUDA static this_type make_formula(formula_ptr&& sub_term) {
    return make<IFormula>(std::move(sub_term));
  }

  CUDA static this_type make_neg(this_ptr&& sub_term) {
    return make<INeg>(Neg(std::move(sub_term)));
  }

  CUDA static this_type make_abs(this_ptr&& sub_term) {
    return make<IAbs>(Abs(std::move(sub_term)));
  }

  CUDA static this_type make_add(this_ptr&& left, this_ptr&& right) {
    return make<IAdd>(Add(std::move(left), std::move(right)));
  }

  CUDA static this_type make_sub(this_ptr&& left, this_ptr&& right) {
    return make<ISub>(Sub(std::move(left), std::move(right)));
  }

  CUDA static this_type make_mul(this_ptr&& left, this_ptr&& right) {
    return make<IMul>(Mul(std::move(left), std::move(right)));
  }

  CUDA static this_type make_tdiv(this_ptr&& left, this_ptr&& right) {
    return make<ITDiv>(TDiv(std::move(left), std::move(right)));
  }

  CUDA static this_type make_fdiv(this_ptr&& left, this_ptr&& right) {
    return make<IFDiv>(FDiv(std::move(left), std::move(right)));
  }

  CUDA static this_type make_cdiv(this_ptr&& left, this_ptr&& right) {
    return make<ICDiv>(CDiv(std::move(left), std::move(right)));
  }

  CUDA static this_type make_ediv(this_ptr&& left, this_ptr&& right) {
    return make<IEDiv>(EDiv(std::move(left), std::move(right)));
  }

  CUDA static this_type make_min(this_ptr&& left, this_ptr&& right) {
    return make<IMin>(Min(std::move(left), std::move(right)));
  }

  CUDA static this_type make_max(this_ptr&& left, this_ptr&& right) {
    return make<IMax>(Max(std::move(left), std::move(right)));
  }

  CUDA static this_type make_naryadd(battery::vector<this_type, allocator_type>&& sub_terms) {
    return make<INaryAdd>(NaryAdd(std::move(sub_terms)));
  }

  CUDA static this_type make_narymul(battery::vector<this_type, allocator_type>&& sub_terms) {
    return make<INaryMul>(NaryMul(std::move(sub_terms)));
  }

  CUDA bool embed(A& a, const U& u) const {
    return forward([&](const auto& t) { return t.embed(a, u); });
  }

  CUDA void project(const A& a, U& r) const {
    return forward([&](const auto& t) { t.project(a, r); });
  }

  CUDA void print(const A& a) const {
    forward([&](const auto& t) { t.print(a); });
  }

  template <class Env, class Allocator2 = typename Env::allocator_type>
  CUDA TFormula<Allocator2> deinterpret(const A& a, const Env& env, AType apc, Allocator2 allocator = Allocator2()) const {
    return forward([&](const auto& t) { return t.deinterpret(a, env, apc, allocator); });
  }

  CUDA size_t length() const {
    return forward([&](const auto& t) { return t.length(); });
  }
};

} // namespace pc
} // namespace lala

#endif
