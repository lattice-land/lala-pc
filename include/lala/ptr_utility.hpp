// Copyright 2022 Pierre Talbot

#ifndef PTR_UTILITY_HPP
#define PTR_UTILITY_HPP

#include "battery/unique_ptr.hpp"
#include "battery/shared_ptr.hpp"

// This function is used to dereference the attribute if T is a unique pointer.
// The rational behind that, is to be able to manipulate a type T as a pointer or a reference.
// In the following code, our term AST is either static (only with template) or dynamic (with virtual function call).
// But we did not want to duplicate the code to handle both.
template <class T>
CUDA const T& deref(const T& x) {
  return x;
}

template <class T>
CUDA T& deref(T& x) {
  return x;
}

template <class T, class Alloc>
CUDA const T& deref(const battery::unique_ptr<T, Alloc>& x) {
  return *x;
}

template <class T, class Alloc>
CUDA T& deref(battery::unique_ptr<T, Alloc>& x) {
  return *x;
}

template <class T, class Alloc>
CUDA const T& deref(const battery::shared_ptr<T, Alloc>& x) {
  return *x;
}

template <class T, class Alloc>
CUDA T& deref(battery::shared_ptr<T, Alloc>& x) {
  return *x;
}

template<class T>
struct remove_ptr { using type = T; };

template<class T, class Alloc>
struct remove_ptr<battery::unique_ptr<T, Alloc>> { using type = T; };

template<class T, class Alloc>
struct remove_ptr<battery::shared_ptr<T, Alloc>> { using type = T; };

#endif
