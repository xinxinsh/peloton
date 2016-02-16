//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// BWTree.h
//
// Identification: src/backend/index/BWTree.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include "backend/index/mapping_table.h"

namespace peloton {
namespace index {

// Look up the stx btree interface for background.
// peloton/third_party/stx/btree.h
template <typename KeyType, typename ValueType, class KeyComparator>
class BWTree {
  typedef uint64_t pid_t;

  // Insertion
  void insert(KeyType key, ValueType value);

  //

 private:
  // TODO: Figure out packing/padding and alignment

  // Abstract base class of all nodes
  struct Node {
    enum NodeType {
      Base,
      Delta,
    };
    // The type
    NodeType node_type;
  };

  // All non-delta nodes are data nodes.  Data nodes contain a high and low
  // fence key, left and right sibling links and a list of keys stored
  // contiguously in this node.
  //
  // There are two types of data nodes, inner nodes store PID links to
  // children (one per key), and leaf nodes that store the values themselves
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

  // Inner nodes
  struct InnerNode : public DataNode {
    // The (contiguous) array of child PID links
    pid_t* pids;
  };

  // Leaf nodes
  struct LeafNode : public DataNode {
    // The (contiguous) array of values
    ValueType* vals;
  }

  // A delta node has a type and a pointer to either the next delta node in the
  // chain, or to the base node. Note that this is an abstract base class for
  // all delta nodes
  struct DeltaNode : public Node {
    enum DeltaType {
      Insert,
      Delete,
      Merge,
      Split,
      Index,
      DeleteIndex,
      RemoveNode
    };
    // The delta type and the next link in the delta chain
    DeltaType type;
    Node* next;
  };

  // A delta insert entry indicates that the stored key and value has been
  // inserted into the logical node
  struct DeltaInsert : public DeltaNode {
    KeyType key;
    ValueType value;
  };

  // A delta delete entry indicates that the stored key has been deleted
  // from the node.
  struct DeltaDelete : public DeltaNode {
    KeyType key;
  };

  // A delta merge entry indicates that the contents of the node pointed to by
  // 'old_right' are now included in this logical node.  For convenience,
  // the smallest key from that node is included here as 'merge_key'
  // TODO: This structure is the same as the split node, code restructure?
  struct DeltaMerge : public DeltaNode {
    KeyType merge_key;
    pid_d new_right;
  };

  // A delta split entry indicates that the contents of the logical node
  // have been split.  The key the split was performed at is stored under
  // the 'key' attribute here, and the PID of the node holding the data
  // whose key is greater than the key in this element is stored under
  // 'new_right'
  struct DeltaSplit : public DeltaNode {
    KeyType split_key;
    pid_d new_right;
  };

  // An index delta indicates that a new index entry was added to this inner
  // node.  We store the key and the child that stores the data
  struct DeltaIndex : public DeltaNode {
    KeyType index_key;
    pid_t child_pid;
  };

  // A simple hash scheme for the mapping table to use
  struct DumbHash {
    std::hash<uint32_t> hash;
    size_t operator()(const pid_t& pid) {
      return hash(pid);
    }
  };

 private:
  // The root of the tree
  pid_t root_;
  // The mapping table
  MappingTable<pid_t, Node*, DumbHash> mapping_table_;
};

}  // End index namespace
}  // End peloton namespace
