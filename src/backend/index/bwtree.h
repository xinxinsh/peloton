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

#include "backend/common/logger.h"
#include "backend/index/epoch_manager.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <stack>
#include <set>
#include <type_traits>
#include <vector>

namespace peloton {
namespace index {

template <typename KeyType, typename ValueType, typename Hasher>
class MappingTable {
 public:
  // Constructor
  MappingTable(): MappingTable(1 << 20) { /* Nothing */ }

  MappingTable(uint32_t initial_size) : size_(initial_size) {
    table_ = new Entry*[initial_size];
    for (uint32_t i = 0; i < initial_size; i++) {
      table_[i] = nullptr;
    }
  }

  // Destructor
  ~MappingTable() {
    if (table_ != nullptr) {
      for (uint32_t i = 0; i < size_; i++) {
        if (table_[i] != nullptr) {
          delete table_[i];
        }
      }
      delete[] table_;
    }
  }

  // Get the mapping for the provided key
  inline ValueType Get(KeyType key) const {
    auto* entry = EntryFor(key);
    return entry == nullptr ? nullptr : entry->val.load();
  }

  // CAS the mapping for the provided key whose expected value is also provided
  inline bool Cas(KeyType key, ValueType old_val, ValueType new_val) {
    auto* entry = EntryFor(key);
    if (entry == nullptr) {
      return false;
    }
    return entry->val.compare_exchange_strong(old_val, new_val);
  }

  // Perform a blind set
  inline bool Insert(KeyType key, ValueType val) {
    auto* curr_entry = EntryFor(key);
    if (curr_entry != nullptr) {
      return false;
    }
    uint64_t hash_value = HashFor(key);
    Entry* entry = new Entry();
    entry->hash = hash_value;
    entry->key = key;
    entry->val.store(val);
    table_[hash_value] = entry;
    return true;
  }

  inline bool Remove(KeyType key) {
    uint64_t slot = HashFor(key) % size_;
    if (table_[slot] == nullptr) {
      return false;
    }
    Entry* e = table_[slot];
    table_[slot] = nullptr;
    delete e;
    return true;
  }

 private:
  // The structures we actually store in the mapping table
  struct Entry {
    uint64_t hash;
    KeyType key;
    std::atomic<ValueType> val;
  };

  // Get the entry for the provided key
  Entry* EntryFor(KeyType& key) const {
    // TODO: Check keys, perform second hash etc.
    uint64_t hash = HashFor(key);
    return table_[hash % size_];
  }

  // Hash a given key
  inline uint64_t HashFor(const KeyType key) const { return key_hasher_(key); }

 private:
  // The mapping table
  uint32_t size_;
  Entry** table_;

  // What we use to hash keys
  Hasher key_hasher_;

 private:
  // No funny business!
  MappingTable(const MappingTable&) = delete;
  MappingTable(MappingTable&&) = delete;
  MappingTable& operator=(const MappingTable&) = delete;
  MappingTable& operator=(MappingTable&&) = delete;
};

//===--------------------------------------------------------------------===//
// The BW-Tree
//===--------------------------------------------------------------------===//
template <typename KeyType, typename ValueType, class KeyComparator,
          class ValueComparator, class KeyEqualityChecker>
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
      Inner = 0,
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
    // The lowest and highest key stored in the subtree rooted at this node
    KeyType low_key;
    KeyType high_key;
    // Left and right sibling links
    pid_t right_link;
    pid_t left_link;
    // The number of entries (keys/values) in this node
    uint32_t num_entries;
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
    // A (continguous) array of keys
    KeyType* keys;
    // The (contiguous) array of child PID links
    pid_t* children;

    ~InnerNode() {
      if (keys != nullptr) {
        delete[] keys;
      }
      if (children != nullptr) {
        delete[] children;
      }
    }

    static InnerNode* Create(KeyType low_key, KeyType high_key, pid_t left_link,
                             pid_t right_link, uint32_t num_entries) {
      InnerNode* inner = new InnerNode();
      inner->node_type = Node::NodeType::Inner;
      inner->low_key = low_key;
      inner->high_key = high_key;
      inner->right_link = right_link;
      inner->left_link = left_link;
      inner->num_entries = num_entries;
      inner->keys = new KeyType[num_entries];
      inner->children = new pid_t[num_entries];
      return inner;
    }
  };

  //===--------------------------------------------------------------------===//
  // Leaf nodes
  //===--------------------------------------------------------------------===//
  struct LeafNode : public DataNode {
    // A (continguous) array of keys
    KeyType* keys;
    // The (contiguous) array of values
    ValueType* vals;

    ~LeafNode() {
      if (keys != nullptr) {
        delete[] keys;
      }
      if (vals != nullptr) {
        delete[] vals;
      }
    }

    static LeafNode* Create(KeyType low_key, KeyType high_key, pid_t left_link,
                            pid_t right_link, uint32_t num_entries) {
      LeafNode* leaf = new LeafNode();
      leaf->node_type = Node::NodeType::Leaf;
      leaf->low_key = low_key;
      leaf->high_key = high_key;
      leaf->left_link = left_link;
      leaf->right_link = right_link;
      leaf->num_entries = num_entries;
      leaf->keys = new KeyType[num_entries];
      leaf->vals = new ValueType[num_entries];
      return leaf;
    }
  };

  //===--------------------------------------------------------------------===//
  // A delta node has a type and a pointer to either the next delta node in the
  // chain, or to the base node. Note that this is an abstract base class for
  // all delta nodes
  //===--------------------------------------------------------------------===//
  struct DeltaNode : public Node {
    Node* next;
    // The total number of entries in the logical root whose delta chain starts
    // at this node
    uint32_t num_entries;
  };

  //===--------------------------------------------------------------------===//
  // A delta insert entry indicates that the stored key and value has been
  // inserted into the logical node
  // TODO: Delta insert and delete are the same structure, refactor?
  //===--------------------------------------------------------------------===//
  struct DeltaInsert : public DeltaNode {
    KeyType key;
    ValueType value;

    static DeltaInsert* Create(KeyType key, ValueType value, Node* next,
                               uint32_t num_entries) {
      DeltaInsert* insert = new DeltaInsert();
      insert->node_type = Node::NodeType::DeltaInsert;
      insert->key = key;
      insert->value = value;
      insert->next = next;
      insert->num_entries = num_entries;
      return insert;
    }
  };

  //===--------------------------------------------------------------------===//
  // A delta delete entry indicates that the stored key has been deleted
  // from the node.
  //===--------------------------------------------------------------------===//
  struct DeltaDelete : public DeltaNode {
    KeyType key;
    ValueType value;
  };

  //===--------------------------------------------------------------------===//
  // A delta merge entry indicates that the contents of the node pointed to by
  // 'old_right' are now included in this logical node.
  //===--------------------------------------------------------------------===//
  struct DeltaMerge : public DeltaNode {
    KeyType merge_key;
    pid_t old_right_pid;
    Node* old_right;
  };

  //===--------------------------------------------------------------------===//
  // A delta split entry indicates that the contents of this logical node
  // have been split into two nodes.  Data whose key <= split_key are in this
  // node. Data whose key > split_key can be found in the right-sibling with
  // pid == new_right.
  //===--------------------------------------------------------------------===//
  struct DeltaSplit : public DeltaNode {
    KeyType split_key;
    KeyType high_key;
    pid_t new_right;

    static DeltaSplit* CreateInner(KeyType split_key, KeyType high_key,
                                   pid_t new_right, uint32_t num_entries,
                                   Node* next) {
      DeltaSplit* split = new DeltaSplit();
      split->node_type = Node::NodeType::DeltaSplitInner;
      split->split_key = split_key;
      split->high_key = high_key;
      split->new_right = new_right;
      split->num_entries = num_entries;
      split->next = next;
      return split;
    }

    static DeltaSplit* CreateLeaf(KeyType split_key, KeyType high_key,
                                  pid_t new_right, uint32_t num_entries,
                                  Node* next) {
      DeltaSplit* split = new DeltaSplit();
      split->node_type = Node::NodeType::DeltaSplit;
      split->split_key = split_key;
      split->high_key = high_key;
      split->new_right = new_right;
      split->num_entries = num_entries;
      split->next = next;
      return split;
    }
  };

  //===--------------------------------------------------------------------===//
  // An index delta indicates that a new index entry 'low_key' was added to this
  // inner node as a result of a split of one of this node's children.
  //
  // This node says that the range of keys between low_key and high_key
  // now belong to the node whose PID is new_child_pid
  //
  // In code:
  // - if low_key < search_key <= high_key:
  //     continue to the new_child_pid node
  // - else:
  //     continue along the current delta chain to the base inner node
  //
  // Refer to 'Parent Update' in Section IV.A of the paper for more details.
  //===--------------------------------------------------------------------===//
  struct DeltaIndex : public DeltaNode {
    // K_P
    KeyType low_key;
    // K_Q
    KeyType high_key;
    pid_t new_child_pid;

    static DeltaIndex* Create(KeyType low_key, KeyType high_key,
                              pid_t new_child_pid, uint32_t num_entries,
                              Node* next) {
      auto* index = new DeltaIndex();
      index->node_type = Node::NodeType::DeltaIndex;
      index->low_key = low_key;
      index->high_key = high_key;
      index->new_child_pid = new_child_pid;
      index->num_entries = num_entries;
      index->next = next;
      return index;
    }
  };

  //===--------------------------------------------------------------------===//
  // A delete index entry delta indicates that two children of this inner node
  // have merged together.  The range of keys covered by low_key and high_key
  // are now owned by the PID 'owner' here.
  //
  // In code:
  // - if low_key < search_key <= high_key:
  //     continue to the 'owner' node
  // - else:
  //     continue along the current delta chain to the base inner node
  //
  // The key-pid pair that is actually removed from the key list is
  // <deleted_key, deleted_pid>
  //
  // Refer to 'Parent Update' in Section IV.B of the paper for more details.
  //===--------------------------------------------------------------------===//
  struct DeltaDeleteIndex : public DeltaNode {
    // The key range covered by the new_owner
    KeyType low_key;
    KeyType high_key;
    pid_t owner;
    KeyType deleted_key;
    pid_t deleted_pid;
  };

  //===--------------------------------------------------------------------===//
  // A simple hash scheme for the mapping table to use
  //===--------------------------------------------------------------------===//
  struct DumbHash {
    std::hash<uint32_t> hash;
    size_t operator()(const pid_t& pid) const { return hash(pid); }
  };

  //===--------------------------------------------------------------------===//
  // The result of a search for a key in an inner node in the tree
  //===--------------------------------------------------------------------===//
  struct FindInnerNodeResult {
    KeyType low_separator;
    KeyType high_separator;
    pid_t child;
    bool needs_consolidation;
  };

  //===--------------------------------------------------------------------===//
  // The result of a search for a key in the tree
  //===--------------------------------------------------------------------===//
  struct FindDataNodeResult {
    // The PID of the leaf node that contains the value
    pid_t node_pid = kInvalidPid;
    // The head (root) of the delta chain (if one exists)
    Node* node = nullptr;
    // The path the search took
    std::vector<std::pair<pid_t, KeyType>> traversal_path;
    // The PID of the first inner node we find during a root-to-leaf traversal
    // that needs to be split
    pid_t node_to_split = kInvalidPid;
    // The PID of the first inner node we find during a root-to-leaf traversal
    // that needs to be consolidated
    pid_t node_to_consolidate = kInvalidPid;
  };

 private:
  /// Key compararison functions and objects

  // Return if lhs < rhs
  inline bool KeyLess(const KeyType& lhs, const KeyType& rhs) const {
    return key_comparator_(lhs, rhs);
  }

  // Return if lhs <= rhs
  inline bool KeyLessEqual(const KeyType& lhs, const KeyType& rhs) const {
    return !key_comparator_(rhs, lhs);
  }

  // Return if lhs > rhs
  inline bool KeyGreater(const KeyType& lhs, const KeyType& rhs) const {
    return key_comparator_(rhs, lhs);
  }

  // Return if lhs >= rhs
  inline bool KeyGreaterEqual(const KeyType& lhs, const KeyType& rhs) const {
    return !key_comparator_(lhs, rhs);
  }

  // Return lhs == rhs
  inline bool KeyEqual(const KeyType& lhs, const KeyType& rhs) const {
    return key_equals_(lhs, rhs);
  }

  // Return lhs == rhs for values
  inline bool ValueEqual(ValueType v1, ValueType v2) const {
    return value_comparator_.Compare(v1, v2);
  }

  //===--------------------------------------------------------------------===//
  // A functor that is able to match a given key-value pair in a collection
  // of std::pair<KeyType, ValueType>
  //===--------------------------------------------------------------------===//
  struct KeyValueEquality {
    // The key and equality checkers
    KeyEqualityChecker key_equals;
    ValueComparator val_comp;
    // The key and value we're searching for
    KeyType key;
    ValueType val;

    bool operator()(const std::pair<KeyType, ValueType>& o) const {
      return key_equals(key, o.first) && val_comp.Compare(val, o.second);
    }
  };

  //===--------------------------------------------------------------------===//
  // A functor that is able to match a given key-pid pair in a collection
  // of std::pair<KeyType, pid_t>
  //===--------------------------------------------------------------------===//
  struct KeyPidEquality {
    KeyEqualityChecker key_equals;
    KeyType key;
    pid_t pid_val;

    bool operator()(const std::pair<KeyType, pid_t>& o) const {
      return key_equals(key, o.first) && pid_val == o.second;
    }
  };

  //===--------------------------------------------------------------------===//
  // A functor that is able to compare only the keys of two
  // std::pair<KeyType, ValueType> types.  We use this to perform binary
  // searches over collections of key-value pairs.
  //===--------------------------------------------------------------------===//
  struct KeyOnlyComparator {
    // The comparator
    KeyComparator cmp;

    bool operator()(const std::pair<KeyType, ValueType>& lhs,
                    const std::pair<KeyType, ValueType>& rhs) const {
      return cmp(lhs.first, rhs.first);
    }

    bool operator()(const std::pair<KeyType, ValueType>& lhs,
                    const KeyType& rhs) const {
      return cmp(lhs.first, rhs);
    }

    bool operator()(const KeyType& lhs,
                    const std::pair<KeyType, ValueType>& rhs) const {
      return cmp(lhs, rhs.first);
    }

    bool operator()(const std::pair<KeyType, pid_t>& lhs,
                    const std::pair<KeyType, pid_t>& rhs) const {
      return cmp(lhs.first, rhs.first);
    }

    bool operator()(const std::pair<KeyType, pid_t>& lhs,
                    const KeyType& rhs) const {
      return cmp(lhs.first, rhs);
    }

    bool operator()(const KeyType& lhs,
                    const std::pair<KeyType, pid_t>& rhs) const {
      return cmp(lhs, rhs.first);
    }
  };

  struct KeyPidComparator {
    KeyComparator cmp;

    bool operator()(const std::pair<KeyType, pid_t>& lhs,
                    const std::pair<KeyType, pid_t>& rhs) const {
      return cmp(lhs.first, rhs.first) && lhs.second < rhs.second;
    }
  };

  //===--------------------------------------------------------------------===//
  // This is the struct that we use when we mark nodes as deleted.  The epoch
  // manager invokes the Free(...) callback when the given data can be deleted.
  // We just call tree.FreeNode(...) with the provided node.
  //===--------------------------------------------------------------------===//
  struct NodeDeleter {
    // The tree
    BWTree<KeyType, ValueType, KeyComparator, ValueComparator,
           KeyEqualityChecker>* tree;
    // The node that can be deleted
    Node* node;

    // Free the captured node's memory
    void Free() { tree->FreeNode(node); }
  };

 public:
  //===--------------------------------------------------------------------===//
  // Our STL iterator over the tree.  We use this guy for both forward and
  // reverse scans
  //===--------------------------------------------------------------------===//
  class BWTreeIterator : public std::iterator<std::input_iterator_tag,
                                              std::pair<KeyType, ValueType>> {
   public:
    BWTreeIterator(
        const BWTree<KeyType, ValueType, KeyComparator, ValueComparator,
                     KeyEqualityChecker>& tree,
        uint32_t curr_idx, pid_t node_pid, Node* node,
        std::vector<std::pair<KeyType, ValueType>>&& collapsed_contents)
        : tree_(tree),
          curr_idx_(curr_idx),
          node_pid_(node_pid),
          node_(node),
          collapsed_contents_(std::move(collapsed_contents)) {
      /* Nothing */
    }

    // Increment
    BWTreeIterator& operator++() {
      // Enter the epoch
      EpochGuard<NodeDeleter> guard{
          const_cast<EpochManager<NodeDeleter>&>(tree_.epoch_manager_)};

      if (curr_idx_ + 1 < collapsed_contents_.size()) {
        curr_idx_++;
      } else {
        // We've exhausted this node, find the next
        pid_t node_pid_ = tree_.RightSibling(node_);
        if (node_pid_ == kInvalidPid) {
          // No more data
          LOG_DEBUG("At right-most leaf, no more data");
          curr_idx_ = 0;
          node_ = nullptr;
          collapsed_contents_.clear();
        } else {
          LOG_DEBUG("Moving right to next tree node [%lu]", node_pid_);
          curr_idx_ = 0;
          node_ = tree_.GetNode(node_pid_);
          collapsed_contents_.clear();
          tree_.CollapseLeafData(node_, collapsed_contents_);
        }
      }
      return *this;
    }

    // Equality/Inequality checks
    bool operator==(const BWTreeIterator& other) const {
      return node_ == other.node_ && curr_idx_ == other.curr_idx_;
    }

    bool operator!=(const BWTreeIterator& other) const {
      return !(*this == other);
    }

    std::pair<KeyType, ValueType> operator*() const {
      return collapsed_contents_[curr_idx_];
    }

    // Access to the key the iterator points to
    KeyType key() const { return collapsed_contents_[curr_idx_].first; }

    // Access to the value the iterator points to
    ValueType data() const { return collapsed_contents_[curr_idx_].second; }

   private:
    const BWTree<KeyType, ValueType, KeyComparator, ValueComparator,
                 KeyEqualityChecker>& tree_;
    uint32_t curr_idx_;
    pid_t node_pid_;
    Node* node_;
    std::vector<std::pair<KeyType, ValueType>> collapsed_contents_;
  };

 public:
  /// *** The public API
  typedef typename BWTree<KeyType, ValueType, KeyComparator, ValueComparator,
                          KeyEqualityChecker>::BWTreeIterator Iterator;
  friend class BWTreeIterator;
  friend class NodeDeleter;

  // Constructor
  BWTree(bool unique_keys, KeyComparator keyComparator,
         ValueComparator valueComparator, KeyEqualityChecker equals)
      : unique_keys_(unique_keys),
        pid_allocator_(0),
        root_pid_(pid_allocator_++),
        leftmost_leaf_pid_(root_pid_.load()),
        rightmost_leaf_pid_(root_pid_.load()),
        key_comparator_(keyComparator),
        value_comparator_(valueComparator),
        key_equals_(equals) {
    // Create a new root page
    LeafNode* root = new LeafNode();
    root->node_type = Node::NodeType::Leaf;
    root->right_link = root->left_link = kInvalidPid;
    root->num_entries = 0;
    root->keys = nullptr;
    root->vals = nullptr;
    // Insert into mapping table
    mapping_table_.Insert(root_pid_, root);

    // Stop epoch management
    epoch_manager_.Init();
  }

  void InstallIndexDelta(const pid_t left_pid, const pid_t right_pid,
                         const KeyType low_key, const KeyType high_key,
                         std::vector<std::pair<pid_t, KeyType>>& traversal) {
    while (true) {
      // Find the correct parent of the left_pid node
      pid_t parent_pid = kInvalidPid;

      if (!traversal.empty()) {
        bool found_left = false;
        for (int32_t i = traversal.size() - 1; i >= 0; i--) {
          if (found_left) {
            if (!IsDeleted(GetNode(traversal[i].first))) {
              parent_pid = traversal[i].first;
              break;
            }
          } else if (traversal[i].first == left_pid) {
            found_left = true;
          }
        }
      }

      // If parent == nullptr, when we're creating a new root
      if (parent_pid == kInvalidPid) {
        LOG_DEBUG("Node [%lu] is the root, we'll be creating a new root!",
                  left_pid);
        auto* new_root =
            InnerNode::Create(low_key, high_key, kInvalidPid, kInvalidPid, 1);
        new_root->keys[0] = low_key;
        new_root->children[0] = left_pid;
        new_root->children[1] = right_pid;

        // Insert new root into mapping table
        pid_t new_root_pid = NextPid();
        bool inserted = mapping_table_.Insert(new_root_pid, new_root);
        assert(inserted);

        // CAS in
        pid_t old_root_pid = root_pid_.load();
        if (root_pid_.compare_exchange_strong(old_root_pid, new_root_pid)) {
          // Successfully CASed in a new root
          LOG_DEBUG(
              "New root [%lu] created with left child [%lu] and right "
              "child [%lu] (i.e., num_entries = %u)",
              new_root_pid, left_pid, right_pid, new_root->num_entries);
          break;
        }

        // CAS failed, try again
        mapping_table_.Remove(new_root_pid);
        delete new_root;

      } else {
        LOG_DEBUG("Parent of [%lu] is [%lu]. We'll install the delta on it.",
                  left_pid, parent_pid);

        // Before inserting the delta, check if some other thread snuck in and
        // inserted the delta index node we intended to
        Node* parent = GetNode(parent_pid);
        assert(!IsLeaf(parent));
        std::vector<std::pair<KeyType, pid_t>> entries;
        CollapseInnerNodeData(parent, entries);

        for (uint32_t i = 0; i < entries.size(); i++) {
          if (KeyEqual(low_key, entries[i].first) &&
              left_pid == entries[i].second) {
            // Someone inserted the delta index term already, we don't have to
            LOG_DEBUG("Another thread inserted the delta index node. Skipping");
            return;
          }
        }

        // Try to insert
        auto* index = DeltaIndex::Create(low_key, high_key, right_pid,
                                         NodeSize(parent) + 1, parent);
        if (mapping_table_.Cas(parent_pid, parent, index)) {
          // Successful Cas
          LOG_DEBUG(
              "Successfully CASed delta index entry into [%lu]. "
              "Previous was (%p), new is (%p)",
              parent_pid, parent, index);
          break;
        }

        // Failed CAS, retry
        LOG_DEBUG("CAS failed to install delta index into [%lu], retrying",
                  parent_pid);
        delete index;
      }
    }
    // Nothing
  }

  //===--------------------------------------------------------------------===//
  // Split the node whose PID is node_pid. It is entirely possible that the
  // node we'd like to split has already been split by another thread. We
  // therefore perform the conditional check to determine split-candidacy
  // before performing the split in a while(true) loop.
  //===--------------------------------------------------------------------===//
  void SplitNode(const pid_t node_pid,
                 std::vector<std::pair<pid_t, KeyType>>& traversal) {
    LOG_DEBUG("SplitNode: Start [%lu] ...", node_pid);

    KeyType split_key;
    KeyType high_key;
    pid_t new_right_pid = kInvalidPid;
    bool we_did_split = false;
    while (true) {
      // Check if all this is necessary
      Node* node = GetNode(node_pid);
      // If the node has been deleted (for some reason, due to merge), we
      // don't do anything
      if (IsDeleted(node)) {
        return;
      }

      // If the node has miraculously reduced in size (due to another thread
      // performing the split), we need to find the split key and new_right_pid
      // to complete the second half of the split logic (delta index insertion
      // at the parent)
      if (NodeSize(node) < insert_branch_factor) {
        Node* tmp = node;
        while (tmp->node_type != Node::NodeType::DeltaSplit &&
               tmp->node_type != Node::NodeType::DeltaSplitInner) {
          assert(tmp->node_type != Node::NodeType::Inner &&
                 tmp->node_type != Node::NodeType::Leaf);
          DeltaNode* delta = static_cast<DeltaNode*>(tmp);
          tmp = delta->next;
        }
        DeltaSplit* split = static_cast<DeltaSplit*>(tmp);
        split_key = split->split_key;
        high_key = split->high_key;
        new_right_pid = split->new_right;
        break;
      }

      // Step 1: Create right-half of split by creating a new inner node
      DataNode* new_right = nullptr;
      if (IsLeaf(node)) {
        // Leaf
        std::vector<std::pair<KeyType, ValueType>> entries;
        std::pair<pid_t, pid_t> links = CollapseLeafData(node, entries);

        uint32_t split_idx = entries.size() / 2;
        uint32_t num_entries = entries.size() - split_idx;
        split_key = entries[split_idx].first;
        auto* new_leaf = LeafNode::Create(split_key, entries.back().first,
                                          node_pid, links.second, num_entries);
        for (uint32_t i = split_idx; i < entries.size(); i++) {
          new_leaf->keys[i - split_idx] = entries[i].first;
          new_leaf->vals[i - split_idx] = entries[i].second;
        }
        high_key = new_leaf->keys[num_entries - 1];
        new_right = new_leaf;
      } else {
        // Inner node
        std::vector<std::pair<KeyType, pid_t>> entries;
        std::pair<pid_t, pid_t> links = CollapseInnerNodeData(node, entries);

        uint32_t split_idx = entries.size() / 2;
        uint32_t num_entries = entries.size() - split_idx;
        split_key = entries[split_idx].first;
        InnerNode* new_inner =
            InnerNode::Create(split_key, entries.back().first, node_pid,
                              links.second, num_entries);
        for (uint32_t i = split_idx; i < entries.size() - 1; i++) {
          new_inner->keys[i - split_idx] = entries[i].first;
          new_inner->children[i - split_idx] = entries[i].second;
        }
        new_inner->children[num_entries - 1] = entries.back().second;

        high_key = new_inner->keys[num_entries - 1];
        new_right = new_inner;
      }

      // Step 2: Insert node holding right-half of split into mapping table
      new_right_pid = NextPid();
      bool inserted = mapping_table_.Insert(new_right_pid, new_right);
      assert(inserted);

      uint32_t new_left_size = NodeSize(node) - NodeSize(new_right);
      LOG_DEBUG(
          "SplitNode: Left-split node [%lu] (%p) has %u keys. "
          "Right-split node [%lu] (%p) has %u keys",
          node_pid, node, new_left_size, new_right_pid, new_right,
          new_right->num_entries);

      // Step 3: Create split delta node and try to CAS it into the left-node
      DeltaSplit* split = nullptr;
      if (IsLeaf(node)) {
        split = DeltaSplit::CreateLeaf(split_key, high_key, new_right_pid,
                                       new_left_size, node);
      } else {
        split = DeltaSplit::CreateInner(split_key, high_key, new_right_pid,
                                        new_left_size, node);
      }
      LOG_DEBUG("Attempting to CAS split-delta [%lu]", node_pid);
      if (mapping_table_.Cas(node_pid, node, split)) {
        // Success, proceed to the next step
        LOG_DEBUG("Successful CAS of split-delta onto node [%lu] (%p)",
                  node_pid, GetNode(node_pid));
        we_did_split = true;

        // Modify the right-most leaf PID if we just created the new right-most
        pid_t curr_rightmost_leaf = rightmost_leaf_pid_.load();
        if (node_pid == curr_rightmost_leaf) {
          LOG_DEBUG("Updating right-most leaf PID from [%lu] to [%lu]",
                    curr_rightmost_leaf, new_right_pid);
          bool changed = rightmost_leaf_pid_.compare_exchange_strong(
              curr_rightmost_leaf, new_right_pid);
          if (!changed) {
            // This is okay. Inserting the split delta makes the right-half
            // externally visible.  In the time that we inserted the delta
            // and we're here, the newly created right-half may have accepted
            // enough inserts to be split multiple times.
          }
        }
        break;
      }

      // Failed, cleanup and try again
      LOG_DEBUG("Failed CAS split-delta onto node [%lu], retrying", node_pid);
      mapping_table_.Remove(new_right_pid);
      delete new_right;
      delete split;
    }

    if (we_did_split) {
      LOG_DEBUG(
          "Installed split-delta node on %lu. Attempting to insert new "
          "split index delta on parent ...",
          node_pid);
    } else {
      LOG_DEBUG(
          "Someone else installed split-delta node on %lu. Attempting to "
          "insert new split index delta on parent ...",
          node_pid);
    }

    // Step 4: Try to install the delta into the parent inner node
    InstallIndexDelta(node_pid, new_right_pid, split_key, high_key, traversal);
  }

  // Insertion
  bool Insert(KeyType key, ValueType value) {
    // Enter the epoch
    EpochGuard<NodeDeleter> guard{epoch_manager_};

    uint32_t attempt = 0;
    while (true) {
      // Find the leaf-level node where we'll insert the data
      FindDataNodeResult result = FindDataNode(key);
      assert(result.node != nullptr);
      assert(IsLeaf(result.node));

      Node* leaf = result.node;
      uint32_t num_entries = NodeSize(leaf);

      // Probe the leaf node to find if this is a duplicate
      if (unique_keys_) {
        auto probe_result = FindInLeafNode(leaf, key, value);
        bool exists = probe_result.first;
        //bool leaf_needs_consolidation = probe_result.second;
        if (exists) {
          LOG_DEBUG("Attempted to insert duplicate key-value pair");
          return false;
        }
      }
      bool leaf_needs_consolidation = NeedsConsolidation(leaf);

      // At this point, everything is okay to insert the delta node for the
      // insertion.  Let's try to CAS in the delta insert node.
      auto* delta_insert =
          DeltaInsert::Create(key, value, leaf, num_entries + 1);
      if (mapping_table_.Cas(result.node_pid, leaf, delta_insert)) {
        // SUCCESS
        LOG_DEBUG("Installed delta insert into node [%lu]", result.node_pid);
        if (num_entries > insert_branch_factor) {
          // Split the leaf we inserted into
          SplitNode(result.node_pid, result.traversal_path);
        }
        // Consolidate we if need to
        if (result.node_to_consolidate != kInvalidPid) {
          ConsolidateNode(result.node_to_consolidate);
        } else if (leaf_needs_consolidation) {
          ConsolidateNode(result.node_pid);
        }
        break;
      } else {
        // The CAS failed, delete the node and retry
        LOG_DEBUG("Failed to CAS in insert delta in attempt %u", ++attempt);
        num_failed_cas_++;
        delete delta_insert;
      }
    }
    return true;
  }

  bool Delete(KeyType key, const ValueType value) {
    // Enter the epoch
    EpochGuard<NodeDeleter> guard{epoch_manager_};

    // Find the leaf-level node where we'll insert the data
    FindDataNodeResult result = FindDataNode(key);
    assert(IsLeaf(result.node));

    Node* prev_root = result.node;
    uint32_t num_entries = NodeSize(prev_root);

    auto probe_result = FindInLeafNode(result.node, key, value);
    bool exists = probe_result.first;
    bool leaf_needs_consolidation = probe_result.second;
    if (!exists) {
      LOG_DEBUG("Attempted to delete non-existent key-value from index");
      return false;
    }

    // TODO: wrap in while loop for multi threaded cases
    auto delta_delete = new DeltaDelete();
    delta_delete->key = key;
    delta_delete->value = value;
    // TODO: make this class into c++ initializer list
    delta_delete->node_type = Node::NodeType::DeltaDelete;
    delta_delete->num_entries = num_entries - 1;
    delta_delete->next = prev_root;

    // Try to CAS
    while (!mapping_table_.Cas(result.node_pid, prev_root, delta_delete)) {
      // TODO: need to retraverse in multithreading
      prev_root = mapping_table_.Get(result.node_pid);
      delta_delete->next = prev_root;
    }
    if (num_entries < delete_branch_factor) {
      // Merge(result.leaf_node, node_pid, deltaInsert);
    }
    if (result.node_to_consolidate != kInvalidPid) {
      ConsolidateNode(result.node_to_consolidate);
    } else if (leaf_needs_consolidation) {
      ConsolidateNode(result.node_pid);
    }
    LOG_DEBUG("Inserted delta delete to node %lu", result.node_pid);
    return true;
  }

  // Return an iterator that points to the element that is associated with the
  // provided key, or an iterator that is equal to what is returned by end()
  Iterator Search(const KeyType key) {
    // Enter the epoch
    EpochGuard<NodeDeleter> guard{epoch_manager_};

    // Find the data node where the key may be
    FindDataNodeResult result = FindDataNode(key);
    assert(result.node != nullptr);
    assert(IsLeaf(result.node));

    // Collapse the chain+delta into a vector of values
    std::vector<std::pair<KeyType, ValueType>> vals;
    CollapseLeafData(result.node, vals);

    // Binary search to find which slot to begin search
    auto found_pos = std::lower_bound(vals.begin(), vals.end(), key,
                                      KeyOnlyComparator{key_comparator_});
    if (found_pos == vals.end()) {
      LOG_DEBUG("Didn't find key in leaf");
      return end();
    }

    uint32_t slot = found_pos - vals.begin();

    LOG_DEBUG("Found key in leaf slot %d", slot);
    if (result.node_to_consolidate != kInvalidPid) {
      ConsolidateNode(result.node_to_consolidate);
    }

    return Iterator{*this, slot, result.node_pid, result.node, std::move(vals)};
  }

  // Get the total size of the tree in bytes. This represents the total amount
  // memory the tree has allocated, including nodes that have been marked for
  // removed but not freed
  uint64_t GetTreeSize() const { return size_in_bytes_.load(); }

  // C++ container iterator functions (hence, why they're not capitalized)
  Iterator begin() const {
    Node* leftmost_leaf = GetNode(leftmost_leaf_pid_);
    std::vector<std::pair<KeyType, ValueType>> vals;
    CollapseLeafData(leftmost_leaf, vals);
    return Iterator{*this, 0, leftmost_leaf_pid_, leftmost_leaf,
                    std::move(vals)};
  }

  Iterator end() const {
    std::vector<std::pair<KeyType, ValueType>> empty;
    return Iterator{*this, 0, kInvalidPid, nullptr, std::move(empty)};
  }

 private:
  // Private APIs

  // Find the leaf-level data node that stores the value associated with the
  // given key. This method is sufficiently general that it can be used for
  // insertions, deletions and scans.  Insertions can call this to get back
  // the pid_t of the leaf that the insertion should take place at.  The
  // pid_t is sufficient to create a delta node and CAS it in.  The traversal
  // path is also available if the node gets deleted while we're CASing in.
  FindDataNodeResult FindDataNode(const KeyType key) {
    // The path we take during the traversal/search
    std::vector<std::pair<pid_t, KeyType>> traversal;
    // The PID of the node that needs consolidation, if any
    pid_t node_to_consolidate = kInvalidPid;
    // The PID of the node we're currently probing
    pid_t curr = root_pid_.load();
    while (true) {
      assert(curr != kInvalidPid);
      Node* curr_node = GetNode(curr);
      assert(curr_node != nullptr);
      if (IsLeaf(curr_node)) {
        // The node we're at is a leaf. If the leaf has been deleted (due to a
        // merge), we have to go back up to the parent and re-traverse to find
        // the correct leaf that has its contents.  Otherwise we're really at a
        // leaf node
        // NOTE: It can be guaranteed that if the node has been deleted, the
        //       last delta (head of the chain) MUST be a DeltaRemoveNode.
        if (!IsDeleted(curr_node)) {
          // We've found the leaf
          LOG_DEBUG("Ending at leaf [%lu] (%p)", curr, curr_node);
          traversal.emplace_back(curr, KeyType());
          break;
        }
        curr = traversal.back().first;
        traversal.pop_back();
      } else {
        // Is an inner node, perform a search for the key in the inner node,
        // return the PID of the next node to go go
        auto search_result = FindInInnerNode(curr, key, traversal);
        pid_t child = search_result.child;

        // Check if the node needs to be consolidated
        bool needs_consolidation = search_result.needs_consolidation;
        if (needs_consolidation && node_to_consolidate == kInvalidPid) {
          node_to_consolidate = curr;
        }

        if (child == kInvalidPid) {
          LOG_DEBUG("Child [%lu] was deleted, going back up tree", child);
          // The inner node was deleted, back up and try again
          curr = traversal.back().first;
          traversal.pop_back();
        } else {
          LOG_DEBUG("Going down to node [%lu] (%p)", child, GetNode(child));
          traversal.emplace_back(curr, search_result.high_separator);
          curr = child;
        }
      }
    }

    // The result
    FindDataNodeResult result;
    result.node_pid = curr;
    result.node = GetNode(curr);
    result.traversal_path = traversal;
    result.node_to_consolidate = node_to_consolidate;
    return result;
  }

  // Find the path to take to the given key in the given inner node
  FindInnerNodeResult FindInInnerNode(
      const pid_t node_pid, const KeyType key,
      std::vector<std::pair<pid_t, KeyType>>& traversal) {
    Node* node = GetNode(node_pid);
    assert(node != nullptr);
    assert(!IsLeaf(node));

    uint32_t chain_length = 0;
    const Node* curr_node = node;
    while (curr_node->node_type != Node::NodeType::Inner) {
      chain_length++;
      switch (curr_node->node_type) {
        case Node::NodeType::DeltaMergeInner: {
          const auto* merge = static_cast<const DeltaMerge*>(curr_node);
          if (KeyLessEqual(key, merge->merge_key)) {
            // key <= merge_key, continue along the current path
            curr_node = merge->next;
          } else {
            // TODO: Complete the SMO
            curr_node = merge->old_right;
          }
          break;
        }
        case Node::NodeType::DeltaSplitInner: {
          const auto* split = static_cast<const DeltaSplit*>(curr_node);
          if (KeyLessEqual(key, split->split_key)) {
            // key <= split_key, continue along the current path
            curr_node = split->next;
          } else {
            // Complete the SMO
            InstallIndexDelta(node_pid, split->new_right, split->split_key,
                              split->high_key, traversal);
            // Go to the right node since key > split_key
            curr_node = GetNode(split->new_right);
          }
          break;
        }
        case Node::NodeType::DeltaIndex: {
          // If delta.low_key < key <= delta.high_key, then we follow the path
          // to the child this index entry points to
          const auto* index = static_cast<const DeltaIndex*>(curr_node);
          if (KeyLess(index->low_key, key) &&
              KeyLessEqual(key, index->high_key)) {
            FindInnerNodeResult result;
            result.low_separator = index->low_key;
            result.high_separator = index->high_key;
            result.child = index->new_child_pid;
            result.needs_consolidation = chain_length >= chain_length_threshold;
            return result;
          } else {
            curr_node = index->next;
          }
          break;
        }
        case Node::NodeType::DeltaDeleteIndex: {
          const auto* del = static_cast<const DeltaDeleteIndex*>(curr_node);
          if (KeyLess(del->low_key, key) && KeyLessEqual(key, del->high_key)) {
            curr_node = GetNode(del->owner);
          } else {
            curr_node = del->next;
          }
          break;
        }
        case Node::NodeType::DeltaRemoveInner: {
          // This node has been deleted (presumably from a merge to another
          // node). Go back up the traversal path and try again
          FindInnerNodeResult result;
          result.low_separator = result.high_separator = KeyType();
          result.child = kInvalidPid;
          result.needs_consolidation = chain_length >= chain_length_threshold;
          return result;
        }
        case Node::NodeType::Leaf: {
          LOG_DEBUG("WTF Leaf?!");
        }
        default: {
          // Anything else should be impossible for inner nodes
          LOG_DEBUG("Hit node type %s on inner node traversal. This is impossible!",
                    std::to_string(curr_node->node_type).c_str());
          assert(false);
        }
      }
    }

    LOG_DEBUG("Chain length for inner-node [%lu] is %u", node_pid, chain_length);

    // Curr now points to the base inner node
    const auto* inner = static_cast<const InnerNode*>(curr_node);
    LOG_DEBUG("Base inner node [%lu] size: %u", node_pid, inner->num_entries);
    auto iter = std::lower_bound(inner->keys, inner->keys + inner->num_entries,
                                 key, key_comparator_);
    uint32_t child_index = iter - inner->keys;

    // Setup the result
    FindInnerNodeResult result;
    if (child_index > 0) {
      result.low_separator = inner->keys[child_index - 1];
    } else {
      result.low_separator = inner->low_key;
    }
    if (child_index < inner->num_entries) {
      result.high_separator = inner->keys[child_index];
    } else {
      result.high_separator = inner->high_key;
    }
    result.child = inner->children[child_index];
    result.needs_consolidation = chain_length >= chain_length_threshold;
    return result;
  }

  // Find the given key in the provided leaf node.
  std::pair<bool, bool> FindInLeafNode(const Node* node, const KeyType key,
                                       const ValueType val) {
    assert(IsLeaf(node));

    uint32_t chain_length = 0;
    const Node* curr = node;
    while (curr->node_type != Node::NodeType::Leaf) {
      switch (curr->node_type) {
        case Node::NodeType::DeltaInsert: {
          // Check if the inserted key is what we're looking for
          const auto* insert = static_cast<const DeltaInsert*>(curr);
          if (KeyEqual(key, insert->key) && ValueEqual(val, insert->value)) {
            return std::make_pair(true, chain_length >= chain_length_threshold);
          }
          curr = insert->next;
          break;
        }
        case Node::NodeType::DeltaDelete: {
          // Check if the key we're looking for has been deleted
          const auto* del = static_cast<const DeltaDelete*>(curr);
          if (KeyEqual(key, del->key) && ValueEqual(val, del->value)) {
            return std::make_pair(false,
                                  chain_length >= chain_length_threshold);
          }
          curr = del->next;
          break;
        }
        case Node::NodeType::DeltaMerge: {
          const auto* merge = static_cast<const DeltaMerge*>(curr);
          if (KeyLess(key, merge->merge_key)) {
            // The key is still in this logical node
            curr = merge->next;
          } else {
            curr = merge->old_right;
          }
          break;
        }
        case Node::NodeType::DeltaSplit: {
          const auto* split = static_cast<const DeltaSplit*>(curr);
          if (KeyLess(key, split->split_key)) {
            curr = split->next;
          } else {
            curr = GetNode(split->new_right);
          }
          break;
        }
        default: {
          LOG_DEBUG("Hit node %s on leaf traversal.  This is impossible!",
                    std::to_string(curr->node_type).c_str());
          assert(false);
        }
      }
      chain_length++;
    }

    LOG_DEBUG("FindInLeafNode: Node chain length is %u", chain_length);

    // We're at the base leaf, just binary search the guy
    const LeafNode* leaf = static_cast<const LeafNode*>(curr);

    // TODO: We need a key-value comparator so that we can use binary search
    auto range = std::equal_range(leaf->keys, leaf->keys + leaf->num_entries,
                                  key, key_comparator_);
    bool found = false;
    for (uint32_t i = range.first - leaf->keys, end = range.second - leaf->keys;
         i < end; i++) {
      if (ValueEqual(val, leaf->vals[i])) {
        found = true;
        break;
      }
    }
    return std::make_pair(found, chain_length >= chain_length_threshold);
  }

  std::pair<pid_t, pid_t> CollapseInnerNodeData(
      Node* node, std::vector<std::pair<KeyType, pid_t>>& output) const {
    assert(node != nullptr);
    assert(!IsLeaf(node));

    std::stack<Node*, std::vector<Node*>> chain;
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Inner) {
      auto* delta = static_cast<DeltaNode*>(curr);
      chain.push(delta);
      curr = delta->next;
    }

    // Curr now points to the base inner
    InnerNode* inner = static_cast<InnerNode*>(curr);
    KeyPidComparator cmp{key_comparator_};

    // Put all inner data into the output vector
    for (uint32_t i = 0; i < inner->num_entries; i++) {
      output.emplace_back(inner->keys[i], inner->children[i]);
    }
    output.emplace_back(KeyType(), inner->children[inner->num_entries]);
    uint32_t base_size = output.size();

    pid_t left_sibling = inner->left_link;
    pid_t right_sibling = inner->right_link;
    uint32_t deleted = 0;
    uint32_t inserted = 0;
    uint32_t chain_length = 0;
    // Process delta chain backwards
    while (!chain.empty()) {
      DeltaNode* delta = static_cast<DeltaNode*>(chain.top());
      chain.pop();
      chain_length++;
      switch (delta->node_type) {
        case Node::NodeType::DeltaIndex: {
          DeltaIndex* index = static_cast<DeltaIndex*>(delta);
          std::pair<KeyType, pid_t> kv{index->low_key, index->new_child_pid};
          auto pos = std::lower_bound(output.begin(), output.end(), kv, cmp);
          output.insert(pos, kv);
          inserted++;
          break;
        }
        case Node::NodeType::DeltaDeleteIndex: {
          DeltaDeleteIndex* del = static_cast<DeltaDeleteIndex*>(delta);
          std::pair<KeyType, pid_t> kv{del->deleted_key, del->deleted_pid};
          auto pos = std::lower_bound(output.begin(), output.end(), kv, cmp);
          output.erase(pos);
          deleted++;
          break;
        }
        case Node::NodeType::DeltaMergeInner: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
          CollapseInnerNodeData(merge->old_right, output);
          break;
        }
        case Node::NodeType::DeltaSplitInner: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          output.erase(output.begin() + split->num_entries, output.end());
          right_sibling = split->new_right;
          break;
        }
        default: {
          LOG_DEBUG("Hit node type %s when collapsing inner node. This is bad.",
                    std::to_string(delta->node_type).c_str());
          assert(false);
        }
      }
    }

    LOG_DEBUG(
        "CollapseInnerData: Found %u inserted, %u deleted, chain length %u, "
        "base page size %u, size after insertions/deletions/splits/merges %lu",
        inserted, deleted, chain_length + 1, base_size - 1, output.size() - 1);
    assert(std::is_sorted(output.begin(), output.end(), cmp));
#ifndef NDEBUG
    // Make sure the output is sorted by keys and contains no duplicate
    // key-value pairs
    if (unique_keys_) {
      for (uint32_t i = 1; i < output.size(); i++) {
        assert(KeyLess(output[i - 1].first, output[i].first) ||
               output[i - 1].second != output[i].second);
      }
    }
#endif

    return std::make_pair(left_sibling, right_sibling);
  }

  /*
  std::pair<pid_t, pid_t> CollapseInnerNodeData(
      Node* node, std::vector<std::pair<KeyType, pid_t>>& output) const {
    assert(node != nullptr);
    assert(!IsLeaf(node));
    uint32_t chain_length = 0;

    std::vector<std::pair<KeyType, pid_t>> inserted;
    std::vector<std::pair<KeyType, pid_t>> deleted;

    // A vector we keep around to collect merged contents from other nodes
    std::vector<std::pair<KeyType, pid_t>> merged;
    pid_t rightmost_merged = kInvalidPid;

    pid_t left_sibling = kInvalidPid;
    pid_t right_sibling = kInvalidPid;
    KeyType* stop_key = nullptr;
    Node* curr = node;
    // Process delta chain backwards
    while (curr->node_type != Node::NodeType::Leaf) {
      switch (curr->node_type) {
        case Node::NodeType::DeltaIndex: {
          DeltaIndex* index = static_cast<DeltaIndex*>(curr);
          // If there was a previous split node and the inserted index is for a
          // key that belongs to the right-half, we don't consider it an
          // entry in this logical node. It is part this node's right sibling
          KeyPidEquality kv_equals{key_equals_, index->low_key,
                                   index->new_child_pid};
          if (stop_key == nullptr || KeyLess(index->low_key, *stop_key)) {
            // Check if low key belongs in inserted index entries (and in the
            // colapsed contents for this node)
            if (std::find_if(inserted.begin(), inserted.end(), kv_equals) ==
                    inserted.end() &&
                std::find_if(deleted.begin(), deleted.end(), kv_equals) ==
                    deleted.end()) {
              inserted.emplace_back(index->low_key, index->new_child_pid);
            }
          }
          curr = index->next;
          break;
        }
        case Node::NodeType::DeltaDeleteIndex: {
          DeltaDeleteIndex* del = static_cast<DeltaDeleteIndex*>(curr);
          // Check if low_key belongs in the collapsed contents
          KeyPidEquality kv_equals{key_equals_, del->deleted_key,
                                   del->deleted_pid};
          if (std::find_if(inserted.begin(), inserted.end(), kv_equals) ==
                  inserted.end() &&
              std::find_if(deleted.begin(), deleted.end(), kv_equals) ==
                  deleted.end()) {
            deleted.emplace_back(del->deleted_key, del->deleted_pid);
          }
          curr = del->next;
          break;
        }
        case Node::NodeType::DeltaMergeInner: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
          CollapseInnerNodeData(merge->old_right, merged);
          curr = merge->next;
          break;
        }
        case Node::NodeType::DeltaSplitInner: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          if (right_sibling == kInvalidPid) {
            stop_key = &split->split_key;
            right_sibling = split->new_right;
          }
          curr = split->next;
          break;
        }
        default: {
          LOG_DEBUG("Hit node type %s when collapsing inner node. This is bad.",
                    std::to_string(curr->node_type).c_str());
          assert(false);
        }
      }
      chain_length++;
    }

    // Curr now points to the base inner node
    InnerNode* inner = static_cast<InnerNode*>(curr);
    KeyOnlyComparator cmp{key_comparator_};

    // Put all leaf data into the output vector
    for (uint32_t i = 0; i < inner->num_entries; i++) {
      output.push_back(std::make_pair(inner->keys[i], inner->children[i]));
    }
    output.push_back(
        std::make_pair(KeyType(), inner->children[inner->num_entries]));

    // Add in merged contents
    for (uint32_t i = 0; i < merged.size(); i++) {
      output.push_back(merged[i]);
    }

    LOG_DEBUG(
        "CollpaseInnerNode: Found %lu inserted, %lu deleted, chain length %d, "
        "base page size %lu",
        inserted.size(), deleted.size(), chain_length + 1, output.size());

    // Check if keys are sorted
    assert(std::is_sorted(output.begin(), output.end(), cmp));

    // Remove all entries that have been split away from this node
    if (stop_key != nullptr) {
      auto pos = std::lower_bound(output.begin(), output.end(), *stop_key, cmp);
      assert(pos != output.end());
      output.erase(pos, output.end());
    }

    // Add inserted key-value pairs from the delta chain
    for (uint32_t i = 0; i < inserted.size(); i++) {
      auto pos =
          std::lower_bound(output.begin(), output.end(), inserted[i], cmp);
      output.insert(pos, inserted[i]);
    }

    // Remove deleted key-value pairs from the delta chain
    for (uint32_t i = 0; i < deleted.size(); i++) {
      auto range =
          std::equal_range(output.begin(), output.end(), deleted[i].first, cmp);
      for (auto pos = range.first, end = range.second; pos != end; ++pos) {
        std::pair<KeyType, pid_t> key_value = *pos;
        if (KeyEqual(key_value.first, deleted[i].first) &&
            key_value.second == deleted[i].second) {
          output.erase(pos);
          break;
        }
      }
    }

    LOG_DEBUG("Final collapsed contents size after deletes: %lu",
              output.size());
#ifndef NDEBUG
    // Make sure the output is sorted by keys and contains no duplicate
    // key-value pairs
    assert(std::is_sorted(output.begin(), output.end(), cmp));
    for (uint32_t i = 1; i < output.size(); i++) {
      assert(key_comparator_(output[i - 1].first, output[i].first) ||
             output[i - 1].second != output[i].second);
    }
#endif
    return std::make_pair(left_sibling, right_sibling);
  }
  */

  std::pair<pid_t, pid_t> CollapseLeafData(
      Node* node, std::vector<std::pair<KeyType, ValueType>>& output) const {
    assert(IsLeaf(node));

    std::stack<Node*, std::vector<Node*>> chain;
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Leaf) {
      auto* delta = static_cast<DeltaNode*>(curr);
      chain.push(delta);
      curr = delta->next;
    }
    uint32_t chain_length = chain.size() + 1;

    // Curr now points to the base leaf
    LeafNode* leaf = static_cast<LeafNode*>(curr);
    KeyOnlyComparator cmp{key_comparator_};

    // Put all leaf data into the output vector
    for (uint32_t i = 0; i < leaf->num_entries; i++) {
      output.push_back(std::make_pair(leaf->keys[i], leaf->vals[i]));
    }
    uint32_t base_size = output.size();

    pid_t left_sibling = leaf->left_link;
    pid_t right_sibling = leaf->right_link;
    uint32_t deleted = 0;
    uint32_t inserted = 0;
    // Process delta chain backwards
    while (!chain.empty()) {
      assert(std::is_sorted(output.begin(), output.end(), cmp));
      DeltaNode* delta = static_cast<DeltaNode*>(chain.top());
      chain.pop();
      switch (delta->node_type) {
        case Node::NodeType::DeltaInsert: {
          DeltaInsert* insert = static_cast<DeltaInsert*>(delta);
          auto pos =
              std::lower_bound(output.begin(), output.end(), insert->key, cmp);
          output.insert(pos, std::make_pair(insert->key, insert->value));
          inserted++;
          break;
        }
        case Node::NodeType::DeltaDelete: {
          DeltaDelete* del = static_cast<DeltaDelete*>(delta);
          auto range = std::equal_range(output.begin(), output.end(), del->key, cmp);
          for (auto iter = range.first, end = range.second;
               iter != end && iter != output.end(); ) {
            std::pair<KeyType, ValueType> key_value = *iter;
            if (KeyEqual(key_value.first, del->key) &&
                ValueEqual(key_value.second, del->value)) {
              iter = output.erase(iter);
              deleted++;
            } else {
              iter++;
            }
          }
          break;
        }
        case Node::NodeType::DeltaMerge: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(delta);
          CollapseLeafData(merge->old_right, output);
          break;
        }
        case Node::NodeType::DeltaSplit: {
          DeltaSplit* split = static_cast<DeltaSplit*>(delta);
          output.erase(output.begin() + split->num_entries, output.end());
          right_sibling = split->new_right;
          break;
        }
        default: {
          LOG_DEBUG("Hit node type %s when collapsing leaf data. This is bad.",
                    std::to_string(curr->node_type).c_str());
          assert(false);
        }
      }
    }
    LOG_DEBUG(
        "CollapseLeafData: Found %u inserted, %u deleted, chain length %u, "
        "base page size %u, size after insertions/deletions/splits/merges %lu",
        inserted, deleted, chain_length, base_size, output.size());
    assert(std::is_sorted(output.begin(), output.end(), cmp));
#ifndef NDEBUG
    // Make sure the output is sorted by keys and contains no duplicate
    // key-value pairs
    if (unique_keys_) {
      for (uint32_t i = 1; i < output.size(); i++) {
        bool equal_keys = KeyEqual(output[i - 1].first, output[i].first);
        bool equal_vals = ValueEqual(output[i - 1].second, output[i].second);
        assert(!(equal_keys && equal_vals));
      }
    }
#endif
    return std::make_pair(left_sibling, right_sibling);
  }

  /*
  std::pair<pid_t, pid_t> CollapseLeafData2(
      Node* node, std::vector<std::pair<KeyType, ValueType>>& output) const {
    assert(node != nullptr);
    assert(IsLeaf(node));
    uint32_t chain_length = 0;
    uint32_t num_entries = NodeSize(node);

    // We use vectors here to track inserted key/value pairs.  Yes, lookups
    // here are O(n), but we don't expect delta chains to be all that long.
    // These should fit nicely in CPU caches making linear lookups pretty
    // fast.
    std::vector<std::pair<KeyType, ValueType>> inserted;
    std::vector<std::pair<KeyType, ValueType>> deleted;

    pid_t left_sibling = kInvalidPid;
    pid_t right_sibling = kInvalidPid;
    //KeyType* stop_key = nullptr;
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Leaf) {
      switch (curr->node_type) {
        case Node::NodeType::DeltaInsert: {
          DeltaInsert* insert = static_cast<DeltaInsert*>(curr);
          // If there was a previous split node and the delta insert is for a
          // key that belongs to the right-half, we don't consider it an
          // entry in this logical node. It is part this node's right sibling
          //if (stop_key == nullptr || key_comparator_(insert->key, *stop_key)) {
            KeyValueEquality kv_equals{key_equals_, value_comparator_,
                                       insert->key, insert->value};
            if (std::find_if(inserted.begin(), inserted.end(), kv_equals) ==
                    inserted.end() &&
                std::find_if(deleted.begin(), deleted.end(), kv_equals) ==
                    deleted.end()) {
              inserted.emplace_back(insert->key, insert->value);
            }
          //}
          curr = insert->next;
          break;
        }
        case Node::NodeType::DeltaDelete: {
          DeltaDelete* del = static_cast<DeltaDelete*>(curr);
          KeyValueEquality kv_equals{key_equals_, value_comparator_, del->key,
                                     del->value};
          // If this deleted key wasn't previously inserted or deleted, we
          // actually need to delete it
          if (std::find_if(inserted.begin(), inserted.end(), kv_equals) ==
                  inserted.end() &&
              std::find_if(deleted.begin(), deleted.end(), kv_equals) ==
                  deleted.end()) {
            deleted.emplace_back(del->key, del->value);
          }
          curr = del->next;
          break;
        }
        case Node::NodeType::DeltaMerge: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
          CollapseLeafData(merge->old_right, output);
          curr = merge->next;
          break;
        }
        case Node::NodeType::DeltaSplit: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          if (right_sibling == kInvalidPid) {
            //stop_key = &split->split_key;
            right_sibling = split->new_right;
          }
          curr = split->next;
          break;
        }
        default: {
          LOG_DEBUG("Hit node type %s when collapsing leaf data. This is bad.",
                    std::to_string(curr->node_type).c_str());
          assert(false);
        }
      }
      chain_length++;
    }

    // Curr now points to a true blue leaf node
    LeafNode* leaf = static_cast<LeafNode*>(curr);
    KeyOnlyComparator cmp{key_comparator_};

    // Put all leaf data into the output vector
    output.clear();
    for (uint32_t i = 0; i < leaf->num_entries; i++) {
      output.push_back(std::make_pair(leaf->keys[i], leaf->vals[i]));
    }

    // Remove all entries that have been split away from this node
    //
    //if (stop_key != nullptr) {
    //  LOG_DEBUG("Collapsed size before removing split data: %lu",
    //            output.size());
    //  auto pos = std::lower_bound(output.begin(), output.end(), *stop_key, cmp);
    //  assert(pos != output.end());
    //  output.erase(pos, output.end());
    //}

    LOG_DEBUG(
        "CollapseLeafData: Found %lu inserted, %lu deleted, chain length %d, "
        "base page size %lu",
        inserted.size(), deleted.size(), chain_length + 1, output.size());

    // Add inserted key-value pairs from the delta chain
    for (uint32_t i = 0; i < inserted.size(); i++) {
      auto pos =
          std::lower_bound(output.begin(), output.end(), inserted[i], cmp);
      output.insert(pos, inserted[i]);
    }

    // Remove deleted key-value pairs from the delta chain
    for (uint32_t i = 0; i < deleted.size(); i++) {
      auto range =
          std::equal_range(output.begin(), output.end(), deleted[i].first, cmp);
      for (auto pos = range.first, end = range.second; pos != end; ++pos) {
        std::pair<KeyType, ValueType> key_value = *pos;
        if (KeyEqual(key_value.first, deleted[i].first) &&
            ValueEqual(key_value.second, deleted[i].second)) {
          output.erase(pos);
          break;
        }
      }
    }

    if (num_entries < output.size()) {
      LOG_DEBUG("Node size is %u, collapsed size is %lu, we'll trim", num_entries, output.size());
      output.erase(output.begin() + num_entries, output.end());
    }

    LOG_DEBUG("Final collapsed contents size = %lu", output.size());

#ifndef NDEBUG
    // Make sure the output is sorted by keys and contains no duplicate
    // key-value pairs
    assert(std::is_sorted(output.begin(), output.end(), cmp));
    for (uint32_t i = 1; i < output.size(); i++) {
      bool equal_keys = KeyEqual(output[i - 1].first, output[i].first);
      bool equal_vals = ValueEqual(output[i - 1].second, output[i].second);
      assert(!(equal_keys && equal_vals));
    }
#endif
    return std::make_pair(left_sibling, right_sibling);
  }
  */

  void ConsolidateNode(pid_t node_pid) {
    Node* curr = GetNode(node_pid);
    if (IsLeaf(curr)) {
      ConsolidateLeafNode(node_pid);
    } else {
      ConsolidateInnerNode(node_pid);
    }
  }

  void ConsolidateInnerNode(pid_t node_pid) {
    uint32_t attempt = 0;
    Node* node = nullptr;
    InnerNode* consolidated = nullptr;
    do {
      // Get the current node
      node = GetNode(node_pid);
      if (IsDeleted(node)) {
        // If someone snuck in and deleted the node before we could consolidate
        // it, then we're really kind of done.  We just mark the node(+chain)
        // to be deleted in this epoch
        // TODO: Mark deleted
        LOG_DEBUG("Looks like inner-node %lu was deleted, not touching ...",
                  node_pid);
        return;
      } else if (!NeedsConsolidation(node)) {
        LOG_DEBUG("Node [%lu] no longer needs consolidation", node_pid);
        return;
      }

      LOG_DEBUG("Consolidating inner-node %lu (%p). Attempt %u", node_pid, node,
                attempt++);

      // Consolidate data
      std::vector<std::pair<KeyType, pid_t>> vals;
      std::pair<pid_t, pid_t> links = CollapseInnerNodeData(node, vals);

      // New leaf node, populate keys and values
      consolidated = InnerNode::Create(vals.front().first, vals.back().first,
                                       links.first, links.second, vals.size());
      for (uint32_t i = 0; i < vals.size() - 1; i++) {
        consolidated->keys[i] = vals[i].first;
        consolidated->children[i] = vals[i].second;
      }
      consolidated->children[vals.size()] = vals.back().second;
    } while (!mapping_table_.Cas(node_pid, node, consolidated));

    // Mark the node as deleted
    assert(node != nullptr);
    epoch_manager_.MarkDeleted(NodeDeleter{this, node});
    LOG_DEBUG("Inner-node %lu consolidation successful, marking for deletion",
              node_pid);
  }

  void ConsolidateLeafNode(pid_t node_pid) {
    uint32_t attempt = 0;
    Node* node = nullptr;
    LeafNode* consolidated = nullptr;
    do {
      // Get the current node
      node = GetNode(node_pid);
      if (IsDeleted(node)) {
        // If someone snuck in and deleted the node before we could consolidate
        // it, then we're really kind of done.  We just mark the node(+chain)
        // to be deleted in this epoch
        // TODO: Mark deleted
        LOG_DEBUG("Looks like leaf-node %lu was deleted, not touching ...",
                  node_pid);
        return;
      } else if (GetChainLength(node) < chain_length_threshold) {
        LOG_DEBUG("Node [%lu] no longer needs consolidation", node_pid);
        return;
      }

      LOG_DEBUG("Consolidating leaf-node %lu (%p). Attempt %u", node_pid, node,
                attempt++);

      // Consolidate data
      std::vector<std::pair<KeyType, ValueType>> vals;
      std::pair<pid_t, pid_t> links = CollapseLeafData(node, vals);

      // New leaf node, populate keys and values
      consolidated = LeafNode::Create(vals.front().first, vals.back().first,
                                      links.first, links.second, vals.size());
      for (uint32_t i = 0; i < vals.size(); i++) {
        consolidated->keys[i] = vals[i].first;
        consolidated->vals[i] = vals[i].second;
      }
    } while (!mapping_table_.Cas(node_pid, node, consolidated));

    // Mark the node as deleted
    assert(node != nullptr);
    epoch_manager_.MarkDeleted(NodeDeleter{this, node});
    LOG_DEBUG("Leaf %lu (%p) consolidation complete, marking for deletion",
              node_pid, node);
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

  // Has this node been deleted?
  bool IsDeleted(const Node* node) const {
    return node->node_type == Node::NodeType::DeltaRemoveLeaf ||
           node->node_type == Node::NodeType::DeltaRemoveInner;
  }

  pid_t RightSibling(const Node* node) const {
    assert(node != nullptr);
    const Node* curr = node;
    while (curr->node_type != Node::NodeType::Inner &&
           curr->node_type != Node::NodeType::Leaf) {
      if (curr->node_type == Node::NodeType::DeltaSplit ||
          curr->node_type == Node::NodeType::DeltaSplitInner) {
        const auto* split = static_cast<const DeltaSplit*>(curr);
        return split->new_right;
      } else {
        const auto* delta = static_cast<const DeltaNode*>(curr);
        curr = delta->next;
      }
    }
    const auto* data_node = static_cast<const DataNode*>(curr);
    return data_node->right_link;
  }

  // Return the delta chain length of the given node
  uint32_t GetChainLength(const Node* node) const {
    // TODO: Put the chain length in the delta itself
    uint32_t chain_length = 0;
    const Node* curr = node;
    while (curr->node_type != Node::NodeType::Inner &&
           curr->node_type != Node::NodeType::Leaf) {
      chain_length++;
      const auto* delta = static_cast<const DeltaNode*>(curr);
      curr = delta->next;
    }
    return chain_length;
  }

  bool NeedsConsolidation(const Node* node) const {
    return GetChainLength(node) >= chain_length_threshold;
  }

  // Return the size (i.e., the number of key-value or key-pid pairs) in this
  // logical node
  uint32_t NodeSize(const Node* node) const {
    if (node->node_type == Node::NodeType::Leaf ||
        node->node_type == Node::NodeType::Inner) {
      return static_cast<const DataNode*>(node)->num_entries;
    } else {
      return static_cast<const DeltaNode*>(node)->num_entries;
    }
  }

  // Delete the memory for this node's delta chain and base node
  void FreeNode(Node* node) {
    LOG_DEBUG("Freeing node (%p)", node);
    uint32_t chain_length = 0;
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Inner &&
           curr->node_type != Node::NodeType::Leaf) {
      if (curr->node_type == Node::NodeType::DeltaMerge) {
        // There is a merge node, we need to free resources of the node that
        // was merged into this node
        DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
        FreeNode(GetNode(merge->old_right_pid));
      }

      // Delete the delta's memory
      DeltaNode* delta = static_cast<DeltaNode*>(curr);
      curr = delta->next;
      delete delta;
      chain_length++;
    }
    auto node_type = curr->node_type;

    // Delete the base node
    delete curr;

    LOG_DEBUG("Freed node (%p) type %d with chain length of %u", node,
              node_type, chain_length);
  }

  // Return the next free/available PID
  // TODO: We need to reclaim PID
  pid_t NextPid() { return pid_allocator_++; }

 private:
  // Does this index support duplicate keys?
  bool unique_keys_;

  // PID allocator
  std::atomic<uint64_t> pid_allocator_;

  // The root of the tree
  std::atomic<pid_t> root_pid_;

  // The left-most and right-mode leaf PIDs.  Tracking these essentially forms
  // a linked list of leaves that we use for scans. The reason the leftmost
  // PID is non-atomic while the rightmost is atomic is because of splits.
  // Splits always create a new right node, leaving the left with the same PID.
  // At index creation time, the leftmost is the root.  Splitting the root
  // creates a new root, but the PID of left-most leaf's PID is the same.
  //
  // Similarily, the right-most leaf PID can change as a result of a split
  // with the creation of a new node holding the values from the right-half
  // of the split.
  pid_t leftmost_leaf_pid_;
  std::atomic<pid_t> rightmost_leaf_pid_;

  // The comparator used for key comparison
  KeyComparator key_comparator_;
  ValueComparator value_comparator_;
  KeyEqualityChecker key_equals_;

  // The mapping table
  MappingTable<pid_t, Node*, DumbHash> mapping_table_;

  // The epoch manager
  EpochManager<NodeDeleter> epoch_manager_;

  // TODO: just a randomly chosen number now...
  uint32_t delete_branch_factor = 100;
  uint32_t insert_branch_factor = 10;
  uint32_t chain_length_threshold = 10;

  std::atomic<uint64_t> num_failed_cas_;
  std::atomic<uint64_t> num_consolidations_;
  std::atomic<uint64_t> size_in_bytes_;
};

}  // End index namespace
}  // End peloton namespace
