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

//===--------------------------------------------------------------------===//
// The BW-Tree
//===--------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator,
          class ValueComparator>
class BWTree {
  typedef uint64_t pid_t;
  static const pid_t kInvalidPid = std::numeric_limits<pid_t>::max();

 private:
  /// First comes all the node structures
  /// TODO: Figure out packing/padding and alignment

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
    uint32_t num_entries;
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

 public:
  //===--------------------------------------------------------------------===//
  // Our STL iterator over the tree.  We use this guy for both forward and
  // reverse scans
  //===--------------------------------------------------------------------===//
  class BWTreeIterator: public std::iterator<std::input_iterator_tag, ValueType> {
   public:
    BWTreeIterator(uint32_t curr_idx, pid_t node_pid, Node* node,
                   const std::vector<ValueType>&& collapsed_contents)
        : curr_idx_(curr_idx),
          node_pid_(node_pid),
          node_(node),
          collapsed_contents_(std::move(collapsed_contents)) {
      /* Nothing */
    }

    // Increment
    BWTreeIterator& operator++() {
      if (curr_idx_ + 1 < collapsed_contents_.size()) {
        curr_idx_++;
      } else {
        // We've exhausted this node, find the next
        Node* curr = node_;
        while (curr->type != Node::NodeType::Leaf) {
          DeltaNode* delta = static_cast<DeltaNode*>(curr);
          curr = curr->next;
        }
        LeafNode* leaf = static_cast<LeafNode*>(curr);
        pid_t right_sibling = leaf->right_link;
        if (right_sibling == kInvalidPid) {
          // No more data
        } else {
          Node* right_node = GetNode(right_sibling);
          collapsed_contents_.clear();
          assert(IsLeaf(right_node));
          CollapseLeafData(right_node, collapsed_contents_);
        }
      }
      return *this;
    }

    // Equality/Inequality checks
    bool operator==(const BWTreeIterator& other) const {
      return node_ == other.node && curr_idx_ == other.curr_idx_;
    }

    bool operator!=(const BWTreeIterator& other) const {
      return !(*this == other);
    }

    // Access to the key the iterator points to
    KeyType key() const;

    // Access to the value the iterator points to
    ValueType data() const;

   private:
    uint32_t curr_idx_;
    pid_t node_pid_;
    Node* node_;
    const std::vector<ValueType> collapsed_contents_;
  };

 public:
  /// *** The public API
  typedef typename BWTree<KeyType, ValueType, KeyComparator>::BWTreeIterator Iterator;
  friend class BWTreeIterator;

  // Constructor
  BWTree(KeyComparator keyComparator, ValueComparator valueComparator)
      : pid_allocator_(0),
        root_pid_(pid_allocator_++),
        key_comparator_(keyComparator),
        value_comparator_(valueComparator) {
    // Create a new root page
    Node* root = new LeafNode();
    mapping_table_.Insert(root_pid_, root);
  }

  // Insertion
  void Insert(KeyType key, ValueType value);

  bool Delete(KeyType key, __attribute__((unused)) const ValueType value) {
    // TODO: replace it with Search
    FindDataNodeResult result = FindDataNode(key);
    if (result.found != true)
      return false;
    Node *prevRoot = result.leaf_node;
    auto num_entries = result.leaf_node->num_entries;
    auto matched = false;
    for (size_t prevRootValItr = 0; prevRootValItr != num_entries;
         ++prevRootValItr) {
      // TODO: do comparison
      //if (leafPrevRoot->vals[prevRootValItr] == value)
        //matched = true;
    }
    if (matched == false)
      return false;

    num_entries--;
    if (result.head != nullptr) {
      auto delta = static_cast<DeltaNode *>(result.head);
      prevRoot = result.head;
      num_entries = delta->num_entries;
    }
    // TODO: wrap in while loop for multi threaded cases
    auto deltaDelete = new DeltaDelete();
    deltaDelete->key = key;
    // TODO: make this class into c++ initializer list
    deltaDelete->node_type = Node::NodeType::DeltaDelete;
    deltaDelete->next = prevRoot;
    pid_t rootPid = result.node_pid;
    while (!mapping_table_.Cas(rootPid, prevRoot, deltaDelete)) {
      //TODO: need to retraverse in multithreading
      prevRoot = mapping_table_.Get(rootPid);
      deltaDelete->next = prevRoot;
    }
    if (num_entries < delete_branch_factor) {
      //Merge(result.leaf_node, node_pid, deltaInsert);
    }
    return true;
  }

  // Return an iterator that points to the element that is associated with the
  // provided key, or an iterator that is equal to what is returned by end()
  Iterator Search(const KeyType key) {
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

  // C++ container iterator functions (hence, why they're not capitalized)
  Iterator begin() const;

  Iterator end() const;

 private:
  // Private APIs

  // Find the leaf-level data node that stores the value associated with the
  // given key. This method is sufficiently general that it can be used for
  // insertions, deletions and scans.  Insertions can call this to get back
  // the pid_t of the leaf that the insertion should take place at.  The
  // pid_t is sufficient to create a delta node and CAS it in.  The traversal
  // path is also available if the node get's deleted while we're CASing in.
  FindDataNodeResult FindDataNode(const KeyType key) const {
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
    FindInLeafNode(result.head, key, result);
    return result;
  }

  // Find the path to take to the given key in the given inner node
  pid_t FindInInnerNode(const Node* node, const KeyType key) const {
    assert(node != nullptr);
    assert(!IsLeaf(node));

    Node* curr_node = (Node *) node;
    while (true) {
      switch (curr_node->node_type) {
        case Node::NodeType::Inner: {
          // At an inner node, do binary search to find child
          InnerNode* inner = static_cast<InnerNode*>(curr_node);
          auto iter = std::lower_bound(
              inner->keys, inner->keys + inner->num_entries, key, key_comparator_);
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
            curr_node = GetNode(merge->new_right);
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
            curr_node = GetNode(split->new_right);
          }
          break;
        }
        case Node::NodeType::DeltaIndex: {
          // If delta.low_key < key <= delta.high_key, then we follow the path
          // to the child this index entry points to
          DeltaIndex* index = static_cast<DeltaIndex*>(curr_node);
          if (key_comparator_(index->low_key, key) < 0 &&
              key_comparator_(key, index->high_key) <= 0) {
            curr_node = GetNode(index->child_pid);
          } else {
            curr_node = index->next;
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
          // TODO: temporarily comment out log message that fails to compile
          //LOG_DEBUG("Hit node %s on inner node traversal.  This is impossible!",
                    //kNodeTypeToString[curr_node->node_type].c_str());
          assert(false);
      }
    }
    // Should never happen
    assert(false);
  }

  // Find the given key in the provided leaf node.  If found, assign the value
  // reference to the value from the leaf and return true. Otherwise, return
  // false if not found in the leaf.
  void FindInLeafNode(const Node* node, const KeyType key, FindDataNodeResult& result) const {
    assert(IsLeaf(node));

    Node* curr = (Node *) node;
    while (true) {
      switch (curr->node_type) {
        case Node::NodeType::Leaf: {
          // A true blue leaf, just binary search this guy
          LeafNode* leaf = static_cast<LeafNode*>(curr);
          auto iter = std::lower_bound(
              leaf->keys, leaf->keys + leaf->num_entries, key, key_comparator_);
          uint32_t index = iter - leaf->keys;
          result.found = index < leaf->num_entries;
          result.slot_idx = index;
          result.delta_node = nullptr;
          result.leaf_node = leaf;
          return;
        }
        case Node::NodeType::DeltaInsert: {
          // Check if the inserted key is what we're looking for
          DeltaInsert* insert = static_cast<DeltaInsert*>(curr);
          if (key_comparator_(key, insert->key) == 0) {
            // The insert was for the key we're looking for
            result.found = true;
            result.slot_idx = 0;
            result.delta_node = insert;
            result.leaf_node = nullptr;
            return;
          }
          curr = insert->next;
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
          curr = del->next;
          break;
        }
        case Node::NodeType::DeltaMerge: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
          if (key_comparator_(key, merge->merge_key) < 0) {
            // The key is still in this logical node
            curr = merge->next;
          } else {
            curr = GetNode(merge->new_right);
          }
          break;
        }
        case Node::NodeType::DeltaSplit: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          if (key_comparator_(key, split->split_key) < 0) {
            curr = split->next;
          } else {
            curr = GetNode(split->new_right);
          }
          break;
        }
        default: {
          // TODO: temporarily comment out log message that fails to compile
          //LOG_DEBUG("Hit node %s on leaf traversal.  This is impossible!",
                    //kNodeTypeToString[curr->node_type].c_str());
          assert(false);
        }
      }
    }
  }

  void CollapseLeafData(const Node* node, std::vector<ValueType>& output) const {
    assert(IsLeaf(node));

    // We use vectors here to track inserted key/value pairs.  Yes, lookups
    // here are O(n), but we don't expect delta chains to be all that long.
    // These should fit nicely in CPU caches making linear lookups pretty
    // fast.
    std::vector<KeyType> inserted_keys;
    std::vector<ValueType> inserted_vals;
    std::vector<KeyType> deleted_keys;

    bool check_stop = false;
    KeyType stop;
    Node* curr = node;
    while (curr->type != Node::NodeType::Leaf) {
      switch (curr->type) {
        case Node::NodeType::DeltaInsert: {
          DeltaInsert* insert = static_cast<DeltaInsert*>(curr);
          if (check_stop && key_comparator_(insert->key, stop) > 0) {
            // There was a split and the key that this delta represents is actually
            // owned by another node.  We don't include it in our results
          } else if (std::find(inserted_keys.begin(), inserted_keys.end(), insert->key) == inserted_keys.end()
              && std::find(deleted_keys.begin(), deleted_keys.end(), insert->key) == deleted_keys.end()) {
            inserted_keys.push_back(insert->key);
            inserted_vals.push_back(insert->val);
          }
          curr = insert->next;
          break;
        }
        case Node::NodeType::DeltaDelete: {
          DeltaDelete* del = static_cast<DeltaDelete*>(curr);
          if (std::find(inserted_keys.begin(), inserted_keys.end(), curr->key) == inserted_keys.end()
              && std::find(deleted_keys.begin(), deleted_keys.end(), curr->key) == deleted_keys.end()) {
            deleted_keys.push_back(curr->key);
          }
          curr = del->next;
          break;
        }
        case Node::NodeType::Merge: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
          CollapseLeafData(GetNode(merge->new_right));
          curr = merge->next;
          break;
        }
        case Node::NodeType::Split: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          check_stop = true;
          stop = split->split_key;
          break;
        }
        default: {
          LOG_DEBUG("Hit node type %s when collapsing leaf data. This is bad.",
                    kNodeTypeToString[curr->type].c_str());
          assert(false);
        }
      }
    }

    // curr now points to a true blue leaf node
    LeafNode* leaf = static_cast<LeafNode*>(curr);
    std::vector<KeyType> all_keys { leaf->keys, leaf->keys + leaf->num_entries };
    std::vector<ValueType> all_vals { leaf->vals, leaf->vals + leaf->num_entries };
    if (check_stop) {
      auto iter = std::lower_bound(all_keys.begin(), all_keys.end(), stop);
      all_keys.erase(iter, all_keys.end());
    }

    // Sort inserted, delete and all_keys/all_vals
    // Perform 3-way merge into output
  }

  // Get the node with the given pid
  Node* GetNode(pid_t node_pid) const { return mapping_table_.Get(node_pid); }

  // Get the root node
  Node* GetRoot() const { return GetNode(root_pid_.load()); }

  // Is the given node a leaf node
  bool IsLeaf(const Node* node) const {
    switch (node->node_type) {
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
      case Node::NodeType::DeltaDeleteIndex:
      case Node::NodeType::DeltaRemoveInner:
        return false;
    }
    return false;
  }

 private:
  // PID allocator
  std::atomic<uint64_t> pid_allocator_;

  // The root of the tree
  std::atomic<pid_t> root_pid_;
  // The comparator used for key comparison
  KeyComparator key_comparator_;
  ValueComparator value_comparator_;
  // The mapping table
  MappingTable<pid_t, Node*, DumbHash> mapping_table_;
  // TODO: just a randomly chosen number now...
  int delete_branch_factor = 100;

  int insert_branch_factor = 500;
};

}  // End index namespace
}  // End peloton namespace
