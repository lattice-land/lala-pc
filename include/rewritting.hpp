// Copyright 2022 Pierre Talbot

#ifndef REWRITTING_HPP
#define REWRITTING_HPP

#include "ast.hpp"

namespace lala {

template <class F>
CUDA thrust::optional<F> negate(const F& f);

/** not(f1 \/ ... \/ fn) --> not(f1) /\ ... /\ not(fn)
    not(f1 /\ ... /\ fn) --> not(f1) \/ ... \/ not(fn) */
template <class F>
CUDA thrust::optional<F> de_morgan_law(Sig sig_neg, const F& f) {
  auto seq = f.seq();
  typename F::Sequence neg_seq(seq.size());
  for(int i = 0; i < f.seq().size(); ++i) {
    auto neg_i = negate(seq[i]);
    if(neg_i.has_value()) {
      neg_seq[i] = *neg_i;
    }
    else {
      return {};
    }
  }
  return F::make_nary(sig_neg, neg_seq, f.type(), f.approx());
}

template <class F>
CUDA thrust::optional<F> negate(const F& f) {
  if(f.is(F::Seq)) {
    Sig neg_sig;
    switch(f.sig()) {
      case LEQ: neg_sig = GT; break;
      case GEQ: neg_sig = LT; break;
      case EQ: neg_sig = NEQ; break;
      case NEQ: neg_sig = EQ; break;
      case LT: neg_sig = GEQ; break;
      case GT: neg_sig = GT; break;
      case AND:
        return de_morgan_law(OR, f);
      case OR:
        return de_morgan_law(AND, f);
      default:
        return {};
    }
    return F::make_nary(neg_sig, f.seq(), f.type(), f.approx());
  }
  return {};
}

template <class F>
CUDA bool is_predicate(const F& f) {
  if(f.is(F::Seq)) {
    switch(f.sig()) {
      case LEQ:
      case GEQ:
      case LT:
      case GT:
      case EQ:
      case NEQ:
        return true;
    }
  }
  return false;
}

CUDA Sig inv(Sig s) {
  switch(s) {
    case LEQ: return GEQ;
    case GEQ: return LEQ;
    case LT:  return GT;
    case GT:  return LT;
    case EQ:  return EQ;
    case NEQ: return NEQ;
    default: assert(false); return s;
  }
}

/** Given a predicate of the form `t <op> u` (e.g., `x + y <= z + 4`), it transforms it into an equivalent predicate of the form `s <op> k` where `k` is a constant (e.g., `x + y - (z + 4) <= 0`.
If the formula is not a predicate, it is returned unchanged. */
template <class F>
CUDA F move_constants_on_rhs(const F& f) {
  if(is_predicate(f) && !f.seq(1).is(F::Z)) {
    AType aty = f.type();
    Approx appx = f.approx();
    if(f.seq(0).is(F::Z)) {
      return F::make_binary(f.seq(1), inv(f.sig()), f.seq(0), aty, appx);
    }
    else {
      return F::make_binary(
        F::make_binary(f.seq(0), SUB, f.seq(1), aty, appx),
        f.sig(),
        F::make_z(0),
        aty, appx);
    }
  }
  return f;
}

} // namespace lala

#endif
