//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// bwtree.h
//
// Identification: src/backend/index/bwtree.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include "backend/index/mapping_table.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <functional>
#include <limits>
#include <map>
#include <stack>
#include <vector>

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, class KeyComparator>
class BWTree {
  typedef uint64_t pid_t;
  static const pid_t kInvalidPid = std::numeric_limits<pid_t>::max();

  // Forward declare so the typedef works
  class BwTreeIterator;

 public:
  typedef typename BWTree<KeyType, ValueType, KeyComparator>::BwTreeIterator
      Iterator;

  // Constructor
  BWTree(KeyComparator comparator);

  // Insertion
  void Insert(KeyType key, ValueType value);

  // Return an iterator that points to the element that is associated with the
  // provided key, or an iterator that is equal to what is returned by end()
  Iterator Search(KeyType key);

  // C++ container iterator functions (hence, why they're not capitalized)
  Iterator begin() const;
  Iterator end() const;

 private:
  // TODO: Figure out packing/padding and alignment

  //===--------------------------------------------------------------------===//
  // Abstract base class of all nodes
  //===--------------------------------------------------------------------===//
  struct Node {
    enum NodeType {
      Inner,
      Leaf,
      DeltaInsert,
      DeltaDelete,
      DeltaMerge,
      DeltaMergeInner,
      DeltaSplit,
      DeltaSplitInner,
      DeltaIndex,
      DeltaDeleteIndex,
      DeltaRemoveLeaf,
      DeltaRemoveInner
    };
    // The type
    NodeType node_type;
  };

  static const std::map<typename Node::NodeType, std::string> kNodeTypeToString;

  //===--------------------------------------------------------------------===//
  // All non-delta nodes are data nodes.  Data nodes contain a high and low
  // fence key, left and right sibling links and a list of keys stored
  // contiguously in this node.
  //
  // There are two types of data nodes, inner nodes store PID links to
  // children (one per key), and leaf nodes that store the values themselves
  //===--------------------------------------------------------------------===//
  struct DataNode : public Node {
    // Low and high fence key
    KeyType low_key;
    KeyType high_key;
    // Left and right sibling links
    pid_t right_link;
    pid_t left_link;
    // The number of entries (keys/values) in this node
    uint32_t num_entries;
    // A (continguous) array of keys
    KeyType* keys;
    // TODO: If sizeof(ValueType) == sizeof(pid_t) we can just use a union:
    // union {
    //   ValueType value;
    //   pid_t child_pid;
    // }* entries;
    // This allows us to unify inner and leaf node types and simplify code
  };

  //===--------------------------------------------------------------------===//
  // Inner nodes
  //===--------------------------------------------------------------------===//
  struct InnerNode : public DataNode {
    // The (contiguous) array of child PID links
    pid_t* children;
  };

  //===--------------------------------------------------------------------===//
  // Leaf nodes
  //===--------------------------------------------------------------------===//
  struct LeafNode : public DataNode {
    // The (contiguous) array of values
    ValueType* vals;
  };

  //===--------------------------------------------------------------------===//
  // A delta node has a type and a pointer to either the next delta node in the
  // chain, or to the base node. Note that this is an abstract base class for
  // all delta nodes
  //===--------------------------------------------------------------------===//
  struct DeltaNode : public Node {
    Node* next;
  };

  //===--------------------------------------------------------------------===//
  // A delta insert entry indicates that the stored key and value has been
  // inserted into the logical node
  //===--------------------------------------------------------------------===//
  struct DeltaInsert : public DeltaNode {
    KeyType key;
    ValueType value;
  };

  //===--------------------------------------------------------------------===//
  // A delta delete entry indicates that the stored key has been deleted
  // from the node.
  //===--------------------------------------------------------------------===//
  struct DeltaDelete : public DeltaNode {
    KeyType key;
  };

  //===--------------------------------------------------------------------===//
  // A delta merge entry indicates that the contents of the node pointed to by
  // 'old_right' are now included in this logical node.  For convenience,
  // the smallest key from that node is included here as 'merge_key'
  // TODO: This structure is the same as the split node, code restructure?
  //===--------------------------------------------------------------------===//
  struct DeltaMerge : public DeltaNode {
    KeyType merge_key;
    pid_t new_right;
  };

  //===--------------------------------------------------------------------===//
  // A delta split entry indicates that the contents of the logical node
  // have been split.  The key the split was performed at is stored under
  // the 'key' attribute here, and the PID of the node holding the data
  // whose key is greater than the key in this element is stored under
  // 'new_right'
  //===--------------------------------------------------------------------===//
  struct DeltaSplit : public DeltaNode {
    KeyType split_key;
    pid_t new_right;
  };

  //===--------------------------------------------------------------------===//
  // An index delta indicates that a new index entry was added to this inner
  // node as a result of a split of one of this node's children.  The low key
  // represents the key that the split was performed on.  The high key is the
  // previous key that would guide searches to the now-split node.  We include
  // both here to quickly determine if a key should go to the newly creted node
  // contianing half the entries from the node that was split.
  // Refer to 'Parent Update' in Section IV.A of the paper for more details.
  //===--------------------------------------------------------------------===//
  struct DeltaIndex : public DeltaNode {
    KeyType low_key;
    KeyType high_key;
    pid_t child_pid;
  };

  //===--------------------------------------------------------------------===//
  // A simple hash scheme for the mapping table to use
  //===--------------------------------------------------------------------===//
  struct DumbHash {
    std::hash<uint32_t> hash;
    size_t operator()(const pid_t& pid) const {
      return hash(pid);
    }
  };

  //===--------------------------------------------------------------------===//
  // The iterator we use for scans
  //===--------------------------------------------------------------------===//
  class BwTreeIterator
      : public std::iterator<std::input_iterator_tag, ValueType> {
   public:
    BwTreeIterator(uint32_t idx, pid_t node_pid, Node* node,
                   const std::vector<ValueType>&& collapsed_contents);

    // Increment
    BwTreeIterator& operator++();
    // Equality/Inequality checks
    BwTreeIterator operator==(const BwTreeIterator& other);
    BwTreeIterator operator!=(const BwTreeIterator& other);
    // Access
    ValueType operator*();

   private:
    uint32_t curr_idx;
    pid_t node_pid;
    Node* node;
    const std::vector<ValueType> collapsed_contents;
  };

  //===--------------------------------------------------------------------===//
  // The result of a search for a key in the tree
  //===--------------------------------------------------------------------===//
  struct FindDataNodeResult {
    // Was a value found
    bool found;
    // The slot to find the value
    uint32_t slot_idx;
    // The PID of the leaf node that contains the value
    pid_t node_pid;
    // The head (root) of the delta chain (if one exists)
    Node* head;
    // The data was either found in the leaf node or a delta node.  Only one of
    // the following two pointers should be non-null if found is true.
    DeltaNode* delta_node;
    LeafNode* leaf_node;
    // The path the search took
    std::stack<pid_t> traversal_path;
  };

 private:
  // Private APIs

  // Find the leaf-level data node that stores the value associated with the
  // given key. This method is sufficiently general that it can be used for
  // insertions, deletions and scans.  Insertions can call this to get back
  // the pid_t of the leaf that the insertion should take place at.  The
  // pid_t is sufficient to create a delta node and CAS it in.  The traversal
  // path is also available if the node get's deleted while we're CASing in.
  FindDataNodeResult FindDataNode(const KeyType key) const;

  // Find the path to take to the given key in the given inner node
  pid_t FindInInnerNode(const Node* node, const KeyType key) const;

  // Find the given key in the provided leaf node.  If found, assign the value
  // reference to the value from the leaf and return true. Otherwise, return
  // false if not found in the leaf.
  void FindInLeafNode(const Node* node, const KeyType key,
                      FindDataNodeResult& res) const;

  // Get the node with the given pid
  Node* GetNode(pid_t node_pid) const { return mapping_table_.Get(node_pid); }

  // Get the root node
  Node* GetRoot() const { return GetNode(root_pid_.load()); }

  // Is the given node a leaf node
  bool IsLeaf(Node* node) const;

 private:
  // PID allocator
  std::atomic<uint64_t> pid_allocator_;

  // The root of the tree
  std::atomic<pid_t> root_pid_;
  // The comparator used for key comparison
  KeyComparator key_comparator_;
  // The mapping table
  MappingTable<pid_t, Node*, DumbHash> mapping_table_;
};

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator>
BWTree<KeyType, ValueType, KeyComparator>::BWTree(KeyComparator key_comparator)
    : pid_allocator_(0),
      root_pid_(pid_allocator_++),
      key_comparator_(key_comparator) {
  // Create a new root page
  Node* root = new LeafNode();
  mapping_table_.Insert(root_pid_, root);
}

//===----------------------------------------------------------------------===//
// FindDataNode
//===----------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::FindDataNodeResult
BWTree<KeyType, ValueType, KeyComparator>::FindDataNode(const KeyType key)
    const {
  // The path we take during the traversal/search
  std::stack<pid_t> traversal;

  pid_t curr = root_pid_.load();
  while (true) {
    assert(curr != kInvalidPid);
    Node* curr_node = GetNode(curr);
    if (IsLeaf(curr_node)) {
      // The node we're at is a leaf. If the leaf has been deleted (due to a
      // merge), we have to go back up to the parent and re-traverse to find
      // the correct leaf that has its contents.  Otherwise we're really at a
      // leaf node
      // NOTE: It can be guaranteed that if the node has been deleted, the
      //       last delta (head of the chain) MUST be a DeltaRemoveNode.
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
        traversal.push(curr);
        curr = child;
      }
    }
  }

  // Probe the leaf
  FindDataNodeResult result;
  result.node_pid = curr;
  result.head = GetNode(curr);
  result.traversal_path = traversal;
  return FindInLeafNode(result.head, key, result);
}

template <typename KeyType, typename ValueType, class KeyComparator>
typename BWTree<KeyType, ValueType, KeyComparator>::pid_t
BWTree<KeyType, ValueType, KeyComparator>::FindInInnerNode(
    const Node* node, const KeyType key) const {
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
      default:
        // Anything else should be impossible for inner nodes
        LOG_DEBUG("Hit node %s on inner node traversal.  This is impossible!",
                  kNodeTypeToString[curr_node->type].c_str());
        assert(false);
    }
  }
  // Should never happen
  assert(false);
}

template <typename KeyType, typename ValueType, class KeyComparator>
void BWTree<KeyType, ValueType, KeyComparator>::FindInLeafNode(
    const Node* node, const KeyType key,
    typename BWTree<KeyType, ValueType, KeyComparator>::FindDataNodeResult&
        result) const {
  assert(IsLeaf(node));

  Node* curr = node;
  while (true) {
    switch (curr->type) {
      case Node::NodeType::Leaf: {
        // A true blue leaf, just binary search this guy
        LeafNode* leaf = static_cast<LeafNode*>(curr);
        auto iter = std::lower_bound(
            leaf->keys, leaf->keys + leaf->num_entries, key,key_comparator_);
        uint32_t index = iter - leaf->keys;
        result.found = index < leaf->num_entries;
        result.slot_idx = index;
        result.delta_node = nullptr;
        result.leaf_node = curr;
        return;
      }
      case Node::NodeType::DeltaInsert: {
        // Check if the inserted key is what we're looking for
        DeltaInsert* insert = static_cast<DeltaInsert*>(curr);
        if (key_comparator_(key, insert->key) == 0) {
          // The insert was for the key we're looking for
          result.found = true;
          result.slot_idx = 0;
          result.delta_node = curr;
          result.leaf_node = nullptr;
          return;
        }
        curr = curr->next;
        break;
      }
      case Node::NodeType::DeltaDelete: {
        // Check if the key we're looking for has been deleted
        DeltaDelete* del = static_cast<DeltaDelete*>(curr);
        if (key_comparator_(key, del->key) == 0) {
          // The key/value was deleted
          result.found = false;
          result.slot_idx = 0;
          result.delta_node = nullptr;
          result.leaf_node = nullptr;
          return;
        }
        curr = curr->next;
        break;
      }
      case Node::NodeType::DeltaMerge: {
        DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
        if (key_comparator_(key, merge->merge_key) < 0) {
          // The key is still in this logical node
          curr = curr->next;
        } else {
          curr = merge->new_right;
        }
        break;
      }
      case Node::NodeType::DeltaSplit: {
        DeltaSplit* split = static_cast<DeltaSplit*>(curr);
        if (key_comparator_(key, split->split_key) < 0) {
          curr = curr->next;
        } else {
          curr = split->new_right;
        }
        break;
      }
      default: {
        LOG_DEBUG("Hit node %s on leaf traversal.  This is impossible!",
                  kNodeTypeToString[curr->type].c_str());
        assert(false);
      }
    }
  }
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
  FindDataNodeResult result = FindDataNode(key);
  Node* prevRoot = result->leaf_node;
  if (result->head) prevRoot = result->head;
  Node* deltaInsert = DeltaInsert{Node.DeltaInsert, prevRoot, key, value};
  pid_t rootPid = result->node_pid;
  while (!mapping_table_.Cas(rootPid, prevRoot, deltaInsert) {
    prevRoot = mapping_table_.Get(rootPid);
    deltaInsert->next = prevRoot;
  }
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
