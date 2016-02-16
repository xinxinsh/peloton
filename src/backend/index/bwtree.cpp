//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// bwtree.cpp
//
// Identification: src/backend/index/bwtree.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "backend/index/bwtree.h"

namespace peloton {
namespace index {

//===----------------------------------------------------------------------===//
// Constructor
template <typename KeyType, typename ValueType, class KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::BWTree()
    : pid_allocator_(0), root_pid_(pid_allocator_++) {
  // Create a new root page
  Node* root = new LeafNode();
  mapping_table_.set(root_pid_, root);
}

//===----------------------------------------------------------------------===//
// Insert
template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::insert(KeyType key, ValueType value) {
  Node* node = mapping_table_.get(root_pid_);
}

}  // End index namespace
}  // End peloton namespace
