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

#include <atomic>
#include <cstdint>
#include <iterator>
#include <functional>
#include <limits>
#include <vector>

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, class KeyComparator>
class BWTree {
  typedef uint64_t pid_t;
  static const pid_t kInvalidPid = std::numeric_limits<pid_t>::max();

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
    size_t operator()(const pid_t& pid) {
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

 public:
  typedef typename BWTree<KeyType, ValueType, KeyComparator>::BwTreeIterator
      Iterator;

  // Constructor
  BWTree();

  // Insertion
  void Insert(KeyType key, ValueType value);

  // Return an iterator that points to the element that is associated with the
  // provided key, or an iterator that is equal to what is returned by end()
  Iterator Search(KeyType key);

  // C++ container iterator functions (hence, why they're not capitalized)
  Iterator begin() const;
  Iterator end() const;

 private:
  //===--------------------------------------------------------------------===//
  // The result of a search for a key in the tree
  //===--------------------------------------------------------------------===//
  struct FindDataNodeResult {
    // Was a value found
    bool found;
    // The PID of the leaf node that contains the value
    pid_t node_pid;
    // The head (root) of the delta chain (if one exists)
    Node* head;
    // The actual data node
    LeafNode* leaf_node;
  };

  // Given a key, perform a search for the node that stores the value fo the key
  FindDataNodeResult FindDataNode(KeyType key) const;

  // Find the path to take to the given key in the given inner node
  pid_t FindInInnerNode(Node* node, KeyType key) const;

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

}  // End index namespace
}  // End peloton namespace
