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

} // namespace lala

#endif
