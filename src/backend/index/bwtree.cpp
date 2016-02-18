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
#include <cassert>
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
  // The path we take during the traversal/search
  std::stack<pid_t> traversal;

  pid_t curr = root_pid_.load();
  traversal.push(curr);
  while (true) {
    assert(curr != kInvalidPid);
    Node* curr_node = GetNode(curr);
    if (IsLeaf(curr_node)) {
      // The current node is a leaf.  If the leaf has been removed (due to merge),
      // we go back up the tree and re-traverse, otherwise we're really at a leaf node
      if (curr_node->node_type != Node::NodeType::DeltaRemoveLeaf) {
        break;
      } else {
        curr = traversal.top();
        traversal.pop();
      }
    } else {
      // Is an inner node, perform a search for the key in the inner node,
      // return the PID of the next node to go go
      pid_t child = FindInInnerNode(curr_node, key);
      if (child == kInvalidPid) {
        // The inner node was deleted, we go back up the traversal path and
        // try again
        curr = traversal.top();
        traversal.pop();
      } else {
        traversal.push(child);
        curr = child;
      }
    }
  }

  /*
  // We're at a base page, meaning that the key does not exist anywhere in
  // the delta chain.  At this point, we can just binary search the node
  // for the index
  LeafNode* leaf = static_cast<LeafNode*>(curr_node);
  auto iter = std::lower_bound(leaf->keys, leaf->keys+leaf->num_entries,
                               key_comparator_);
  bool found = (iter - leaf->keys) < leaf->num_entries;
  return FindDataNodeResult{found, traversal.top(), nullptr, leaf};
  */
}

template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::pid_t
BWTree<KeyType, ValueType, KeyComparator>::FindInInnerNode(Node* node,
                                                           KeyType key) const {
  assert(node != nullptr);
  assert(!IsLeaf(node));

  Node* curr_node = node;
  while (true) {
    switch (curr_node->type) {
      case Node::NodeType::Inner: {
        // At an inner node, do binary search to find child
        InnerNode* inner = static_cast<InnerNode*>(curr_node);
        auto iter = std::lower_bound(
            inner->keys, inner->keys + inner->num_entries, key_comparator_);
        uint32_t child_index = iter - inner->keys;
        return inner->children[std::min(child_index, inner->num_entries - 1)];
      }
      case Node::NodeType::DeltaMergeInner: {
        // This node has contents that have been merged from another node.
        // Figure out if we need to continue along this logical node or
        // go node containing the newly merged contents
        DeltaMerge* merge = static_cast<DeltaMerge*>(curr_node);
        if (key_comparator_(key, merge->merge_key) <= 0) {
          curr_node = merge->next;
        } else {
          curr_node = merge->new_right;
        }
        break;
      }
      case Node::NodeType::DeltaSplitInner: {
        // This node has been split.  Check to see if we need to go
        // right or left
        DeltaSplit* split = static_cast<DeltaSplit*>(curr_node);
        if (key_comparator_(key, split->split_key) <= 0) {
          curr_node = split->next;
        } else {
          curr_node = split->new_right;
        }
        break;
      }
      case Node::NodeType::DelaIndex: {
        // If delta.low_key < key <= delta.high_key, then we follow the path
        // to the child this index entry points to
        DeltaIndex* index = static_cast<DeltaIndex*>(curr_node);
        if (key_comparator_(index->low_key, key) < 0 &&
            key_comparator_(key, index->high_key) <= 0) {
          curr_node = GetNode(index->child_pid);
        } else {
          curr_node = curr_node->next;
        }
        break;
      }
      case Node::NodeType::DeltaDeleteIndex: {
        break;
      }
      case Node::NodeType::DeltaRemoveInner: {
        // This node has been deleted (presumably from a merge to another node)
        // Go back up the traversal path and try again
        return kInvalidPid;
      }
    }
  }
  // Should never happen
  assert(false);
}

template <typename KeyType, typename ValueType, class KeyComparator>
bool BWTree<KeyType, ValueType, KeyComparator>::IsLeaf(Node* node) const {
  switch (node->type) {
    case Node::NodeType::Leaf:
    case Node::NodeType::DeltaInsert:
    case Node::NodeType::DeltaDelete:
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
