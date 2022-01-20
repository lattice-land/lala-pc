// Copyright 2021 Pierre Talbot

#ifndef IPC_HPP
#define IPC_HPP

#include "ast.hpp"

namespace lala {

template <typename A, typename Alloc, typename Shape>
class IPC {
  DArray<Shape, Alloc> propagators;
};

}

#endif
