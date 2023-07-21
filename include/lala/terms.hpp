// Copyright 2021 Pierre Talbot

#ifndef TERMS_HPP
#define TERMS_HPP

#include "battery/vector.hpp"
#include "lala/universes/primitive_upset.hpp"

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

  CUDA void tell(A&, const U&, local::BInc&) const {}
  CUDA U project(const A&) const { return k; }
  CUDA local::BInc is_top(const A&) const { return false; }
  CUDA void print(const A&) const { ::battery::print(k); }
  template <class Alloc>
  CUDA TFormula<Alloc> deinterpret(const Alloc& alloc, AType apc) const {
    return k.template deinterpret<TFormula<Alloc>>();
  }

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

  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    a.tell(avar, u, has_changed);
  }

  CUDA U project(const A& a) const {
    return a.project(avar);
  }

  CUDA local::BInc is_top(const A& a) const { return project(a).is_top(); }
  CUDA void print(const A& a) const { printf("(%d,%d)", avar.aty(), avar.vid()); }

  template <class Alloc>
  CUDA TFormula<Alloc> deinterpret(const Alloc&, AType) const {
    return TFormula<Alloc>::make_avar(avar);
  }

  template <class A2>
  friend class Variable;
};

template<class Universe>
struct NegOp {
  using U = Universe;

  CUDA static U op(const U& a) {
    return U::template fun<NEG>(a);
  }

  CUDA static U inv(const U& a) {
    return op(a); // negation is its own inverse.
  }

  static constexpr bool function_symbol = false;
  CUDA static const char* symbol() { return "-"; }
  CUDA static Sig sig() { return NEG; }
};

template<class Universe>
struct AbsOp {
  using U = Universe;

  CUDA static U op(const U& a) {
    return U::template fun<ABS>(a);
  }

  CUDA static U inv(const U& a) {
    return meet(a, U::template fun<NEG>(a));
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

  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    if(x().is_top(a)) { return; }
    x().tell(a, UnaryOp::inv(u), has_changed);
  }

  CUDA U project(const A& a) const {
    return UnaryOp::op(x().project(a));
  }

  CUDA local::BInc is_top(const A& a) const {
    return x().is_top(a);
  }

  CUDA void print(const A& a) const {
    printf("%s", UnaryOp::symbol());
    if constexpr(UnaryOp::function_symbol) { printf("("); }
    x().print(a);
    if constexpr(UnaryOp::function_symbol) { printf(")"); }
  }

  template <class Alloc>
  CUDA TFormula<Alloc> deinterpret(const Alloc& allocator, AType apc) const {
    return TFormula<Alloc>::make_unary(UnaryOp::sig(), x().deinterpret(allocator, apc), apc, allocator);
  }
};

template<class Universe>
struct GroupAdd {
  using U = Universe;
  CUDA static U op(const U& a, const U& b) {
    return U::template fun<ADD>(a, b);
  }

  CUDA static U rev_op(const U& a, const U& b) {
    return op(a, U::additive_inverse(b));
  }

  CUDA static U inv1(const U& a, const U& b) {
    return U::template fun<SUB>(a, b);
  }

  CUDA static U inv2(const U& a, const U& b) {
    return inv1(a,b);
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '+'; }
  CUDA static Sig sig() { return ADD; }
};

template<class Universe>
struct GroupSub {
  using U = Universe;
  CUDA static U op(const U& a, const U& b) {
    return U::template fun<SUB>(a, b);
  }

  CUDA static U inv1(const U& a, const U& b) {
    return U::template fun<ADD>(a, b);
  }

  CUDA static U inv2(const U& a, const U& b) {
    return U::template fun<NEG>(
      U::template fun<SUB>(a, b));
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '-'; }
  CUDA static Sig sig() { return SUB; }
};

template<class Universe, Sig divsig>
struct GroupMul {
  using U = Universe;
  CUDA static U op(const U& a, const U& b) {
    return U::template fun<MUL>(a, b);
  }

  CUDA static U rev_op(const U& a, const U& b) {
    return U::template fun<divsig>(a, b); // probably not ideal? Could think more about that.
  }

  CUDA static U inv1(const U& a, const U& b) {
    return U::template fun<divsig>(a, b);
  }

  CUDA static U inv2(const U& a, const U& b) {
    return inv1(a, b);
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '*'; }
  CUDA static Sig sig() { return MUL; }
};

template<class Universe, Sig divsig>
struct GroupDiv {
  using U = Universe;
  CUDA static U op(const U& a, const U& b) {
    return U::template fun<divsig>(a, b);
  }

  CUDA static U inv1(const U& a, const U& b) {
    return U::template fun<MUL>(a, b);
  }

  CUDA static U inv2(const U& a, const U& b) {
    return U::template fun<divsig>(b, a);
  }

  static constexpr bool prefix_symbol = false;
  CUDA static char symbol() { return '/'; }
  CUDA static Sig sig() { return divsig; }
};

template<class Universe, Sig msig>
struct GroupMinMax {
  static_assert(msig == MIN || msig == MAX);

  using U = Universe;
  CUDA static U op(const U& a, const U& b) {
    return U::template fun<msig>(a, b);
  }

  CUDA static U inv1(const U& a, const U& b) {
    if(join(a, b).is_top()) {
      return a;
    }
    else {
      return U::bot();
    }
  }

  CUDA static U inv2(const U& a, const U& b) {
    return inv1(a, b);
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

  /** Enforce `x <op> y >= u` where >= is the lattice order of the underlying abstract universe.
      For instance, over the interval abstract universe, `x + y >= [2..5]` will ensure that `x + y` is eventually at least `2` and at most `5`. */
  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    auto xt = x().project(a);
    auto yt = y().project(a);
    if(xt.is_top() || yt.is_top()) { return; }
    if(!x().is(sub_type::IConstant)) {
      x().tell(a, G::inv1(u, yt), has_changed);   // x <- u <inv> y
    }
    if(!y().is(sub_type::IConstant)) {
      y().tell(a, G::inv2(u, x().project(a)), has_changed);   // y <- u <inv> x
    }
  }

  CUDA U project(const A& a) const {
    return G::op(x().project(a), y().project(a));
  }

  CUDA local::BInc is_top(const A& a) const {
    return x().is_top(a) || y().is_top(a);
  }

  CUDA void print(const A& a) const {
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

  template <class Alloc>
  CUDA TFormula<Alloc> deinterpret(const Alloc& allocator, AType apc) const {
    return TFormula<Alloc>::make_binary(
      x().deinterpret(allocator, apc),
      G::sig(),
      y().deinterpret(allocator, apc),
      apc,
      allocator);
  }
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

  CUDA U project(const A& a) const {
    U accu = t(0).project(a);
    for(int i = 1; i < terms.size(); ++i) {
      accu = G::op(accu, t(i).project(a));
    }
    return accu;
  }

  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    U all = project(a);
    for(int i = 0; i < terms.size(); ++i) {
      t(i).tell(a, G::inv1(u, G::rev_op(all, t(i).project(a))), has_changed);
    }
  }

  CUDA local::BInc is_top(const A& a) const {
    for(int i = 0; i < terms.size(); ++i) {
      if(t(i).is_top(a)) {
        return local::BInc::top();
      }
    }
    return local::BInc::bot();
  }

  CUDA void print(const A& a) const {
    t(0).print(a);
    for(int i = 1; i < terms.size(); ++i) {
      printf(" %c ", G::symbol());
      t(i).print(a);
    }
  }

  template <class Alloc>
  CUDA TFormula<Alloc> deinterpret(const Alloc& alloc, AType apc) const {
    using F = TFormula<Alloc>;
    typename F::Sequence seq{alloc};
    for(int i = 0; i < terms.size(); ++i) {
      seq.push_back(t(i).deinterpret(alloc, apc));
    }
    return TFormula<Alloc>::make_nary(G::sig(), std::move(seq), apc);
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
  CUDA static VTerm create(const Term<A2, Alloc2>& other, const allocator_type& allocator) {
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
  CUDA auto forward(F&& f) const {
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

  CUDA void tell(A& a, const U& u, local::BInc& has_changed) const {
    forward([&](const auto& t) { t.tell(a, u, has_changed); });
  }

  CUDA U project(const A& a) const {
    return forward([&](const auto& t) { return t.project(a); });
  }

  CUDA local::BInc is_top(const A& a) const {
    return forward([&](const auto& t) { return t.is_top(a); });
  }

  CUDA void print(const A& a) const {
    forward([&](const auto& t) { return t.print(a); });
  }

  template <class Alloc>
  CUDA TFormula<Alloc> deinterpret(const Alloc& alloc, AType apc) const {
    return forward([&](const auto& t) { return t.deinterpret(alloc, apc); });
  }
};

} // namespace pc
} // namespace lala

#endif
