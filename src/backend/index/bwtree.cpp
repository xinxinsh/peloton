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

#include <algorithm>
#include <stack>

namespace peloton {
namespace index {

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::BWTree()
    : pid_allocator_(0), root_pid_(pid_allocator_++) {
  // Create a new root page
  Node* root = new LeafNode();
  mapping_table_.set(root_pid_, root);
}


//===----------------------------------------------------------------------===//
// FindDataNode
//===----------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::FindDataNodeResult
BWTree<KeyType, ValueType, KeyComparator>::FindDataNode(KeyType key) const {
  std::stack<pid_t> traversal;
  pid_t curr = root_pid_.load();
  while (true) {
    traversal.push(curr);
    Node* curr_node = GetNode(curr);
    if (curr_node->node_type == Node::NodeType::Leaf) {
      // We're at a base page, meaning that the key does not exist anywhere in
      // the delta chain.  At this point, we can just binary search the node
      // for the index
      LeafNode* leaf = static_cast<LeafNode*>(curr_node);
      auto iter = std::lower_bound(leaf->keys, leaf->keys+leaf->num_entries, key_comparator_);
      bool found = (iter - leaf->keys) < leaf->num_entries;
      return FindDataNodeResult{found, traversal.top(), nullptr, leaf};
    } else {
      // A delta chain
      DeltaNode* delta = static_cast<DeltaNode*>(curr_node);
      switch (delta->type) {
        case DeltaNode::DeltaType::Insert: {
          // Check if the insert is for the key we're interested in
          DeltaInsert* delta_insert = static_cast<DeltaInsert*>(curr_node);
          if (key_comparator_(key, delta_insert->key) == 0) {
            //
          }
        }
      }
    }
  }
  // Not possible
}


template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::IsLeaf(Node* node) const {
  switch (node->type) {
    case Node::NodeType::DeltaInsert:
    case Node::NodeType::DeltaDelete:
    case Node::NodeType::Leaf:
    case Node::NodeType::DeltaMerge:
    case Node::NodeType::DeltaSplit:
    case Node::NodeType::DeltaRemoveLeaf:
      return true;
    case Node::NodeType::Inner:
    case Node::NodeType::DeltaMergeInner:
    case Node::NodeType::DeltaSplitInner:
    case Node::NodeType::DeltaIndex:
    case Node::Nodetype::DeltaDeleteIndex:
    case Node::NodeType::DeltaRemoveInner:
      return false;
  }
}

//===----------------------------------------------------------------------===//
// Insert
//===----------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::Insert(KeyType key,
                                                       ValueType value) {
  // TODO: Fill me out
}

//===----------------------------------------------------------------------===//
// Search
//===----------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::Iterator
BWTree<KeyType, ValueType, KeyComparator>::Search(KeyType key) {
  FindDataNodeResult result = FindDataNode(key);
  if (result.node == nullptr) {
    // The key doesn't exist, return an invalid iterator
    return end();
  }

  // Collapse the chain+delta into a vector of values
  std::vector<ValueType> vals;
  Node* node = result.node;
  if (node->node_type == Node::NodeType::Delta) {

  }
  return Iterator{0, result.node, vals};
}

}  // End index namespace
}  // End peloton namespace
