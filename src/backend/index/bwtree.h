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
#include "backend/index/index_key.h"

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
  ValueType Get(KeyType key) const {
    auto* entry = EntryFor(key);
    return entry == nullptr ? nullptr : entry->val.load();
  }

  // CAS in the mapping for the provided key whose expected value is also provided
  bool Cas(KeyType key, ValueType old_val, ValueType new_val) {
    auto* entry = EntryFor(key);
    if (entry == nullptr) {
      return false;
    }
    return entry->val.compare_exchange_strong(old_val, new_val);
  }

  // Perform a blind set
  bool Insert(KeyType key, ValueType val) {
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
    return table_[hash];
  }

  // Hash a given key
  uint64_t HashFor(const KeyType key) const { return key_hasher_(key); }

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

    ~DataNode() {
      if (keys != nullptr) {
        delete[] keys;
      }
    }
  };

  //===--------------------------------------------------------------------===//
  // Inner nodes
  //===--------------------------------------------------------------------===//
  struct InnerNode : public DataNode {
    // The (contiguous) array of child PID links
    pid_t* children;

    ~InnerNode() {
      if (children != nullptr) {
        delete[] children;
      }
    }

    static InnerNode* Create(KeyType low_key, KeyType high_key,
                             pid_t right_link, pid_t left_link,
                             uint32_t num_entries) {
      InnerNode* inner = new InnerNode();
      inner->node_type = Node::NodeType::Inner;
      inner->low_key = low_key;
      inner->high_key = high_key;
      inner->right_link = right_link;
      inner->left_link = left_link;
      // number of key entries
      inner->num_entries = num_entries;
      inner->keys = new KeyType[num_entries];
      inner->children = new pid_t[num_entries + 1];
      return inner;
    }
  };

  //===--------------------------------------------------------------------===//
  // Leaf nodes
  //===--------------------------------------------------------------------===//
  struct LeafNode : public DataNode {
    // The (contiguous) array of values
    ValueType* vals;

    ~LeafNode() {
      if (vals != nullptr) {
        delete[] vals;
      }
    }

    static LeafNode* Create(KeyType& low_key, KeyType& high_key,
                            pid_t right_link, pid_t left_link,
                            uint32_t num_entries) {
      LeafNode* leaf = new LeafNode();
      leaf->node_type = Node::NodeType::Leaf;
      leaf->low_key = low_key;
      leaf->high_key = high_key;
      leaf->right_link = right_link;
      leaf->left_link = left_link;
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
    // TODO: determine if this needs to be here (needed bc of casting in insert)
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
  // 'old_right' are now included in this logical node.  For convenience,
  // the smallest key from that node is included here as 'merge_key'
  // TODO: This structure is the same as the split node, refactor?
  //===--------------------------------------------------------------------===//
  struct DeltaMerge : public DeltaNode {
    KeyType merge_key;
    Node* old_right;
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
    // TODO: fix this by refactoring or sth
    // Zero for DeltaSplitInner
    // num_entries is sizeo f the original branch (ie left)
    // For SMO completions
    KeyType low_key;
    KeyType high_key;
    pid_t new_child_pid;
    pid_t orig_left_pid;
    bool is_rightmost;
  };

  //===--------------------------------------------------------------------===//
  // An index delta indicates that a new index entry 'low_key' was added to this
  // inner node as a result of a split of one of this node's children.
  //
  // This node says that the range of keys between low_key and high_key
  // now belong to the node whose PID is new_child_pid
  //
  // In code:
  // - if low_key < search_key < high_key:
  //     continue along the child_pid node
  // - else:
  //     continue along this delta chain to the base inner node
  //
  // Refer to 'Parent Update' in Section IV.A of the paper for more details.
  //===--------------------------------------------------------------------===//
  struct DeltaIndex : public DeltaNode {
    KeyType low_key;
    KeyType high_key;
    pid_t new_child_pid;
    bool is_rightmost;
  };

  //===--------------------------------------------------------------------===//
  // A delete index entry delta indicates that a child has been merged into
  // a sibling and the new owner of the key range from 'low_key' to 'high_key'
  // is 'new_owner'.
  //
  // Refer to 'Parent Update' in Section IV.A of the paper for more details.
  //===--------------------------------------------------------------------===//
  struct DeltaDeleteIndex : public DeltaNode {
    KeyType low_key;
    KeyType high_key;
    pid_t new_owner;
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
  //TODO: SplitInner should happen here
  struct FindDataNodeResult {
    // The high key that directed the search to the node
    KeyType high_key;
    // The PID of the leaf node that contains the value
    pid_t node_pid = kInvalidPid;
    // The head (root) of the delta chain (if one exists)
    Node* node = nullptr;
    // The path the search took
    std::stack<pid_t> traversal_path;
    std::stack<KeyType> parent_search_keys;
    // The PID of the first node we find while traversal that needs consolidation
    pid_t node_to_consolidate = kInvalidPid;
    pid_t last_inner_node = kInvalidPid;
    pid_t inner_node_to_split = kInvalidPid;
  };

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

    bool operator()(const std::pair<KeyType, ValueType>& first,
                    const std::pair<KeyType, ValueType>& second) const {
      return cmp(first.first, second.first);
    }

    bool operator()(const std::pair<KeyType, ValueType>& lhs,
                    const KeyType& rhs) const {
      return cmp(lhs.first, rhs);
    }

    bool operator()(const KeyType& lhs,
                    const std::pair<KeyType, ValueType>& rhs) const {
      return cmp(lhs, rhs.first);
    }

    bool operator()(const std::pair<KeyType, pid_t>& first,
                    const std::pair<KeyType, pid_t>& second) const {
      return cmp(first.first, second.first);
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

  //===--------------------------------------------------------------------===//
  // This is the struct that we use when we mark nodes as deleted.  The epoch
  // manager invokes the Free(...) callback when the given data can be deleted.
  // We just call tree.FreeNode(...) with the provided node.
  //===--------------------------------------------------------------------===//
  struct NodeDeleter {
    // The tree
    BWTree<KeyType, ValueType, KeyComparator, ValueComparator, KeyEqualityChecker>* tree;
    // The node that can be deleted
    Node* node;

    void Free() {
      tree->FreeNode(node);
    }
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
      EpochGuard<NodeDeleter> guard{const_cast<EpochManager<NodeDeleter>&>(tree_.epoch_manager_)};

      if (curr_idx_ + 1 < collapsed_contents_.size()) {
        curr_idx_++;
      } else {
        // We've exhausted this node, find the next
        Node* curr = node_;
        while (curr->node_type != Node::NodeType::Leaf) {
          DeltaNode* delta = static_cast<DeltaNode*>(curr);
          curr = delta->next;
        }
        LeafNode* leaf = static_cast<LeafNode*>(curr);
        pid_t right_sibling = leaf->right_link;
        if (right_sibling == kInvalidPid) {
          // No more data
          curr_idx_ = 0;
          node_pid_ = kInvalidPid;
          node_ = nullptr;
          collapsed_contents_.clear();
        } else {
          Node* right_node = tree_.GetNode(right_sibling);
          collapsed_contents_.clear();
          assert(tree_.IsLeaf(right_node));
          tree_.CollapseLeafData(right_node, collapsed_contents_);
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
    KeyType key() const {
      return collapsed_contents_[curr_idx_].first;
    }

    // Access to the value the iterator points to
    ValueType data() const {
      // TODO: Bounds check
      return collapsed_contents_[curr_idx_].second;
    }

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
  typedef typename std::multiset<std::pair<KeyType, ValueType>> KVMultiset;
  friend class BWTreeIterator;
  friend class NodeDeleter;

  // Constructor
  BWTree(KeyComparator keyComparator, ValueComparator valueComparator,
         KeyEqualityChecker equals)
      : pid_allocator_(0),
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

  // Insertion
  bool Insert(KeyType key, ValueType value) {
    // Enter the epoch
    EpochGuard<NodeDeleter> guard{epoch_manager_};
    
    Node* prev_root;
    pid_t node_pid;
    std::stack<pid_t> traversal_path;
    std::stack<KeyType> parent_search_keys;
    pid_t node_to_consolidate = kInvalidPid;
    uint32_t num_entries = 0;
    DeltaInsert* delta_insert;
    bool leaf_needs_consolidation = false;
    pid_t inner_node_to_split = kInvalidPid;
    do {
      // Find the leaf-level node where we'll insert the data
      FindDataNodeResult result = FindDataNode(key);
      node_to_consolidate = result.node_to_consolidate;
      inner_node_to_split = result.inner_node_to_split;
      traversal_path = result.traversal_path;
      parent_search_keys = result.parent_search_keys;
      assert(result.node != nullptr);
      assert(IsLeaf(result.node));

      prev_root = result.node;
      node_pid = result.node_pid;
      
      bool exists;
      pid_t new_split_pid;
      Node* new_split_node;
      std::tie(exists, leaf_needs_consolidation, new_split_pid, new_split_node) = FindInLeafNode(result.node, key, value);
    
      if (exists) {
        LOG_DEBUG("Attempted to insert duplicate key-value pair");
        return false;
      }

      if (new_split_pid != kInvalidPid) {
        prev_root = new_split_node;
        node_pid = new_split_pid;
      }
      
      if (prev_root->node_type == Node::NodeType::Leaf) {
        num_entries = static_cast<LeafNode*>(prev_root)->num_entries + 1;
      } else {
        num_entries = static_cast<DeltaNode*>(prev_root)->num_entries + 1;
      }

      // TODO If remove, goto left sibling and insert there in multithreading case
      delta_insert = new DeltaInsert();
      delta_insert->key = key;
      delta_insert->value = value;
      // TODO: make this class into c++ initializer list
      delta_insert->node_type = Node::NodeType::DeltaInsert;
      delta_insert->next = prev_root;
      delta_insert->num_entries = num_entries;

    } while (!mapping_table_.Cas(node_pid, prev_root, delta_insert));
    if (num_entries > insert_branch_factor) {
      LOG_DEBUG("Splitting since num_entries: %d for node %lu", num_entries, node_pid);
      Split(node_pid, parent_search_keys, false);
    }
    if (node_to_consolidate != kInvalidPid) {
      ConsolidateNode(node_to_consolidate);
    } else if (leaf_needs_consolidation) {
      ConsolidateNode(node_pid);
    }

    if (inner_node_to_split != kInvalidPid) {
      LOG_DEBUG("Splitting inner node %lu", inner_node_to_split);
      Split(inner_node_to_split, parent_search_keys, true);
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
    uint32_t num_entries = 0;
    if (result.node->node_type == Node::NodeType::Leaf) {
      num_entries = static_cast<LeafNode*>(result.node)->num_entries - 1;
    } else {
      num_entries = static_cast<DeltaNode*>(result.node)->num_entries - 1;
    }

    auto probe_result = FindInLeafNode(result.node, key, value);
    bool exists = std::get<0>(probe_result);
    bool leaf_needs_consolidation = std::get<1>(probe_result);
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
    assert(IsLeaf(result.node));

    // Collapse the chain+delta into a vector of values
    std::vector<std::pair<KeyType, ValueType>> vals;
    CollapseLeafData(result.node, vals);

    // Binary search to find which slot to begin search
    auto found_pos = std::lower_bound(vals.begin(), vals.end(), key,
                                      KeyOnlyComparator{key_comparator_});
    if (found_pos == vals.end()) {
      return end();
    }

    uint32_t slot = found_pos - vals.begin();

    LOG_DEBUG("Found key in leaf slot %d", slot);
    if (result.node_to_consolidate != kInvalidPid) {
      ConsolidateNode(result.node_to_consolidate);
    }

    return Iterator{*this, slot, result.node_pid, result.node, std::move(vals)};
  }

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
  // path is also available if the node get's deleted while we're CASing in.
  FindDataNodeResult FindDataNode(const KeyType key) {
    // The path we take during the traversal/search
    std::stack<pid_t> traversal;
    std::stack<KeyType> parent_search_keys;
    // The PID of the node that needs consolidation, if any
    pid_t node_to_consolidate = kInvalidPid;
    // The PID of the node we're currently probing
    pid_t curr = root_pid_.load();
    pid_t inner_node_to_split = kInvalidPid;
    bool is_rightmost;
    pid_t last_inner_node = kInvalidPid;
    while (true) {
      assert(curr != kInvalidPid);
      LOG_DEBUG("curr is %lu", curr);
      Node* curr_node = GetNode(curr);
      if (IsLeaf(curr_node)) {
        // The node we're at is a leaf. If the leaf has been deleted (due to a
        // merge), we have to go back up to the parent and re-traverse to find
        // the correct leaf that has its contents.  Otherwise we're really at a
        // leaf node
        // NOTE: It can be guaranteed that if the node has been deleted, the
        //       last delta (head of the chain) MUST be a DeltaRemoveNode.
        if (!IsDeleted(curr_node)) {
          // We've found the leaf
          break;
        }
        curr = traversal.top();
        traversal.pop();
      } else {
        // Is an inner node, perform a search for the key in the inner node,
        // return the PID of the next node to go go
        last_inner_node = curr;
        pid_t child;
        bool needs_consolidation;
        KeyType inner_search_key;
        bool needs_split;
        std::tie(child, needs_consolidation, inner_search_key, is_rightmost, needs_split) = FindInInnerNode(curr_node, key);

        if (needs_consolidation && node_to_consolidate == kInvalidPid) {
          node_to_consolidate = curr;
        }
        if (needs_split && inner_node_to_split == kInvalidPid) {
          inner_node_to_split = curr;
        }
        if (child == kInvalidPid) {
          // The inner node was deleted, back up and try again
          curr = traversal.top();
          traversal.pop();
        } else {
          if (!is_rightmost) parent_search_keys.push(inner_search_key);
          traversal.push(curr);
          curr = child;
        }
      }
    }

    // The result
    FindDataNodeResult result;
    result.node_pid = curr;
    result.node = GetNode(curr);
    result.traversal_path = traversal;
    result.parent_search_keys = parent_search_keys;
    result.node_to_consolidate = node_to_consolidate;
    result.last_inner_node = last_inner_node;
    result.inner_node_to_split = inner_node_to_split;
    return result;
  }

  // Find the path to take to the given key in the given inner node
  std::tuple<pid_t, bool, KeyType, bool, bool> FindInInnerNode(const Node* node, const KeyType key) const {
    assert(node != nullptr);
    assert(!IsLeaf(node));

    uint32_t chain_length = 0;
    Node* curr_node = (Node *) node;
    KeyType parent_high_key;
    bool is_rightmost = false;
    uint32_t num_entries = curr_node->node_type == Node::NodeType::Inner ? 
      static_cast<InnerNode*>(curr_node)->num_entries : 
      static_cast<DeltaNode*>(curr_node)->num_entries;
    while (curr_node->node_type != Node::NodeType::Inner) {
      switch (curr_node->node_type) {
        case Node::NodeType::DeltaMergeInner: {
          // This node has contents that have been merged from another node.
          // Figure out if we need to continue along this logical node or
          // go node containing the newly merged contents
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr_node);
          if (key_comparator_(key, merge->merge_key)) {
            curr_node = merge->next;
          } else {
            curr_node = merge->old_right;
          }
          break;
        }
        case Node::NodeType::DeltaSplitInner: {
          // This node has been split.  Check to see if we need to go
          // right or left
          DeltaSplit* split = static_cast<DeltaSplit*>(curr_node);
          if (key_comparator_(key, split->split_key)) {
            curr_node = split->next;
          } else {
            //TODO: Post delta for parent
            curr_node = GetNode(split->new_right);
          }
          break;
        }
        case Node::NodeType::DeltaIndex: {
          // If delta.low_key < key <= delta.high_key, then we follow the path
          // to the child this index entry points to
          // TODO: This case might need to be mirrored for deletion
          DeltaIndex* index = static_cast<DeltaIndex*>(curr_node);
          if (key_comparator_(index->low_key, key) &&
              key_comparator_(key, index->high_key)) {
            Node* new_child_node = GetNode(index->new_child_pid);
            curr_node = GetNode(index->new_child_pid);
            if (!IsLeaf(new_child_node)) {
              parent_high_key = index->high_key;
            } else {
              return std::make_tuple(index->new_child_pid, chain_length >= chain_length_threshold, index->high_key, index->is_rightmost, num_entries >= insert_branch_factor);
            }
          } else {
            curr_node = index->next;
          }
          break;
        }
        case Node::NodeType::DeltaDeleteIndex: {
          DeltaDeleteIndex* del = static_cast<DeltaDeleteIndex*>(curr_node);
          if (key_comparator_(del->low_key, key) &&
              key_comparator_(key, del->high_key)) {
            curr_node = GetNode(del->new_owner);
          } else {
            curr_node = del->next;
          }
          break;
        }
        case Node::NodeType::DeltaRemoveInner: {
          // This node has been deleted (presumably from a merge to another
          // node). Go back up the traversal path and try again
          pid_t deleted = kInvalidPid;
          return std::make_tuple(deleted, chain_length >= chain_length_threshold,
              parent_high_key, is_rightmost, num_entries >= insert_branch_factor);
        }
        default: {
          // Anything else should be impossible for inner nodes
          LOG_DEBUG("Hit node %s on inner node traversal. This is impossible!",
                    std::to_string(curr_node->node_type).c_str());
          assert(false);
        }
      }
      chain_length++;
    }

    LOG_DEBUG("Chain length for inner-node was %u", chain_length);

    // Curr now points to the base inner node
    InnerNode* inner = static_cast<InnerNode*>(curr_node);
    auto iter = std::lower_bound(inner->keys, inner->keys + inner->num_entries,
                                 key, key_comparator_);
    uint32_t child_index = iter - inner->keys;
    pid_t result_pid = inner->children[child_index];
    if (result_pid < inner->num_entries + 1) {
      //non-rightmost key
      parent_high_key = inner->keys[child_index];
    } else {
      //rightmost key
      is_rightmost = true;
    }
    return std::make_tuple(result_pid, chain_length >= chain_length_threshold,
        parent_high_key, is_rightmost, num_entries >= insert_branch_factor);
  }

  // Find the given key in the provided leaf node.  If found, assign the value
  // reference to the value from the leaf and return true. Otherwise, return
  // false if not found in the leaf.
  std::tuple<bool, bool, pid_t, Node*> FindInLeafNode(const Node* node,
      const KeyType key, const ValueType val) {
    assert(IsLeaf(node));
    pid_t split_pid = kInvalidPid;
    Node* split_node = NULL;
    uint32_t chain_length = 0;
    Node* curr = (Node *) node;
    //TODO: where's the case for LeafDelete?
    while (curr->node_type != Node::NodeType::Leaf) {
      switch (curr->node_type) {
        case Node::NodeType::DeltaInsert: {
          // Check if the inserted key is what we're looking for
          DeltaInsert* insert = static_cast<DeltaInsert*>(curr);
          if (key_equals_(key, insert->key) &&
              value_comparator_.Compare(val, insert->value)) {
            return std::make_tuple(true, chain_length >= chain_length_threshold,
                split_pid, split_node);
          }
          curr = insert->next;
          break;
        }
        case Node::NodeType::DeltaDelete: {
          // Check if the key we're looking for has been deleted
          DeltaDelete* del = static_cast<DeltaDelete*>(curr);
          if (key_equals_(key, del->key) &&
              value_comparator_.Compare(val, del->value)) {
            return std::make_tuple(false, chain_length >= chain_length_threshold,
                split_pid, split_node);
          }
          curr = del->next;
          break;
        }
        case Node::NodeType::DeltaMerge: {
          DeltaMerge* merge = static_cast<DeltaMerge*>(curr);
          if (key_comparator_(key, merge->merge_key)) {
            // The key is still in this logical node
            curr = merge->next;
          } else {
            curr = merge->old_right;
          }
          break;
        }
        case Node::NodeType::DeltaSplit: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          if (key_comparator_(key, split->split_key)) {
            curr = split->next;
          } else {
            //TODO: SMO completion here?
            // check for child pid_t. if its there, don't install update
            DeltaIndex* delta_index = new DeltaIndex();
            delta_index->low_key = split->low_key;
            delta_index->high_key = split->high_key;
            delta_index->new_child_pid = split->new_child_pid;
            delta_index->is_rightmost = split->is_rightmost;
            delta_index->node_type = Node::NodeType::DeltaSplitInner;
            InstallDeltaIndex(split->orig_left_pid, delta_index, false);
            split_pid = split->new_right;
            curr = GetNode(split->new_right);
            split_node = curr; 
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
    LeafNode* leaf = static_cast<LeafNode*>(curr);

    // TODO: We need a key-value comparator so that we can use binary search
    auto range = std::equal_range(leaf->keys, leaf->keys + leaf->num_entries,
                                  key, key_comparator_);
    bool found = false;
    for (uint32_t i = range.first - leaf->keys, end = range.second - leaf->keys;
         i < end; i++) {
      if (value_comparator_.Compare(val, leaf->vals[i])) {
        found = true;
        break;
      }
    }
    return std::make_tuple(found, chain_length >= chain_length_threshold,
        split_pid, split_node);
  }

  std::pair<pid_t, pid_t> CollapseInnerNodeData(
        Node* node, std::vector<std::pair<KeyType, pid_t>>& output) const {
    assert(node != nullptr);
    assert(!IsLeaf(node));
    uint32_t chain_length = 0;

    std::vector<std::pair<KeyType, pid_t>> inserted;
    std::vector<std::pair<KeyType, pid_t>> deleted;

    // A vector we keep around to collect merged contents from other nodes
    std::vector<std::pair<KeyType, pid_t>> merged;

    pid_t left_sibling = kInvalidPid;
    pid_t right_sibling = kInvalidPid;
    KeyType* stop_key = nullptr;
    Node* curr = node;
    LOG_DEBUG("curr %p", curr);
    while (curr->node_type != Node::NodeType::Inner) {
      switch (curr->node_type) {
        case Node::NodeType::DeltaIndex: {
          DeltaIndex* index = static_cast<DeltaIndex*>(curr);
          // If there was a previous split node and the inserted index is for a
          // key that belongs to the right-half, we don't consider it an
          // entry in this logical node. It is part this node's right sibling
          if (stop_key == nullptr || key_comparator_(*stop_key, index->low_key)) {
            // Check if low key belongs in inserted index entries (and in the
            // colapsed contents for this node)
            KeyPidEquality low_kv_equals{key_equals_, index->low_key, index->new_child_pid};
            if (std::find_if(inserted.begin(), inserted.end(), low_kv_equals) == inserted.end() &&
                std::find_if(deleted.begin(), deleted.end(), low_kv_equals) == deleted.end()) {
              inserted.push_back(std::make_pair(index->low_key, index->new_child_pid));
            }
            // Now check the high key
            KeyPidEquality high_kv_equals{key_equals_, index->high_key, index->new_child_pid};
            if (std::find_if(inserted.begin(), inserted.end(), high_kv_equals) == inserted.end() &&
                std::find_if(deleted.begin(), deleted.end(), high_kv_equals) == deleted.end()) {
              inserted.push_back(std::make_pair(index->high_key, index->new_child_pid));
            }
          }
          curr = index->next;
          break;
        }
        case Node::NodeType::DeltaDeleteIndex: {
          DeltaDeleteIndex* del = static_cast<DeltaDeleteIndex*>(curr);
          // Check if low_key belongs in the collapsed contents
          KeyPidEquality low_kv_equals{key_equals_, del->low_key, del->new_owner};
          if (std::find_if(inserted.begin(), inserted.end(), low_kv_equals) == inserted.end() &&
              std::find_if(deleted.begin(), deleted.end(), low_kv_equals) == deleted.end()) {
            deleted.push_back(std::make_pair(del->low_key, del->new_owner));
          }
          // Check if low_key belongs in the collapsed contents
          KeyPidEquality high_kv_equals{key_equals_, del->high_key, del->new_owner};
          if (std::find_if(inserted.begin(), inserted.end(), high_kv_equals) == inserted.end() &&
              std::find_if(deleted.begin(), deleted.end(), high_kv_equals) == deleted.end()) {
            deleted.push_back(std::make_pair(del->high_key, del->new_owner));
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
        case Node::NodeType::DeltaSplit: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          stop_key = &split->split_key;
          if (right_sibling == kInvalidPid) {
            right_sibling = split->new_right;
          }
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


    // Curr now points to the base inner node
    InnerNode* inner = static_cast<InnerNode*>(curr);
    KeyOnlyComparator cmp{key_comparator_};

    // Put all leaf data into the output vector
    for (uint32_t i = 0; i < inner->num_entries; i++) {
      output.push_back(std::make_pair(inner->keys[i], inner->children[i]));
    }
    for (uint32_t i = 0; i < merged.size(); i++) {
      output.push_back(merged[i]);
    }

    LOG_DEBUG(
        "CollpaseInnerNode: Found %lu inserted, %lu deleted, chain length %d, "
        "base page size %lu" ,
        inserted.size(), deleted.size(), chain_length + 1, output.size());

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
        if (EqualKeys(key_value.first, deleted[i].first) &&
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
      assert(key_comparator_(output[i-1].first, output[i].first) ||
             output[i-1].second != output[i].second);
    }
#endif
    return std::make_pair(left_sibling, right_sibling);
  }
  
  std::pair<pid_t, pid_t> CollapseLeafData(
        Node* node, std::vector<std::pair<KeyType, ValueType>>& output) const {
    assert(node != nullptr);
    assert(IsLeaf(node));
    uint32_t chain_length = 0;

    // We use vectors here to track inserted key/value pairs.  Yes, lookups
    // here are O(n), but we don't expect delta chains to be all that long.
    // These should fit nicely in CPU caches making linear lookups pretty
    // fast.
    std::vector<std::pair<KeyType, ValueType>> inserted;
    std::vector<std::pair<KeyType, ValueType>> deleted;

    pid_t left_sibling = kInvalidPid;
    pid_t right_sibling = kInvalidPid;
    KeyType* stop_key = nullptr;
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Leaf) {
      switch (curr->node_type) {
        case Node::NodeType::DeltaInsert: {
          DeltaInsert* insert = static_cast<DeltaInsert*>(curr);
          // If there was a previous split node and the delta insert is for a
          // key that belongs to the right-half, we don't consider it an
          // entry in this logical node. It is part this node's right sibling
          if (stop_key == nullptr || key_comparator_(insert->key, *stop_key)) {
            KeyValueEquality kv_equals{key_equals_, value_comparator_,
                                       insert->key, insert->value};
            if (std::find_if(inserted.begin(), inserted.end(), kv_equals) ==
                  inserted.end() &&
                std::find_if(deleted.begin(), deleted.end(), kv_equals) ==
                    deleted.end()) {
              inserted.push_back(std::make_pair(insert->key, insert->value));
            }
          }
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
            deleted.push_back(std::make_pair(del->key, del->value));
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
          // TODO: check if this needs to be initiated only once
          stop_key = &split->split_key;
          if (right_sibling == kInvalidPid) {
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
    if (stop_key != nullptr) {
      LOG_DEBUG("Collapsed size before removing split data: %lu",
                output.size());
      auto pos = std::lower_bound(output.begin(), output.end(), *stop_key, cmp);
      assert(pos != output.end());
      output.erase(pos, output.end());
    }

    LOG_DEBUG(
        "CollapseLeafData: Found %lu inserted, %lu deleted, chain length %d, "
        "base page size %lu" ,
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
        if (EqualKeys(key_value.first, deleted[i].first) &&
            EqualValues(key_value.second, deleted[i].second)) {
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
      assert(key_comparator_(output[i-1].first, output[i].first) ||
             !value_comparator_.Compare(output[i-1].second, output[i].second));
    }
#endif
    return std::make_pair(left_sibling, right_sibling);
  }

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
      }

      LOG_DEBUG("Consolidating inner-node %lu (%p). Attempt %u",
                node_pid, node, attempt++);

      // Consolidate data
      std::vector<std::pair<KeyType, pid_t>> vals;
      std::pair<pid_t, pid_t> links = CollapseInnerNodeData(node, vals);

      // New leaf node, populate keys and values
      consolidated = InnerNode::Create(vals.front().first, vals.back().first,
                                       links.first, links.second, vals.size());
      for (uint32_t i = 0; i < vals.size(); i++) {
        consolidated->keys[i] = vals[i].first;
        consolidated->children[i] = vals[i].second;
      }
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
      }

      LOG_DEBUG("Consolidating leaf-node %lu (%p). Attempt %u",
                node_pid, node, attempt++);

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

  InnerNode* GetInner(Node* node) {
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Inner) {
      DeltaNode* delta = static_cast<DeltaNode*>(curr);
      curr = delta->next;
    }
    return static_cast<InnerNode*>(curr);
  }


  LeafNode* GetLeaf(Node* node) {
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Leaf) {
      DeltaNode* delta = static_cast<DeltaNode*>(curr);
      curr = delta->next;
    }
    return static_cast<LeafNode*>(curr);
  }

  void Split(pid_t node_pid, std::stack<KeyType>& parent_search_keys, 
      bool is_inner) {
    Node* node = nullptr;
    KeyType split_key;
    pid_t sibling_pid;
    LOG_DEBUG("Split called with node %lu", node_pid);
    do {
      //TODO: add delete logic for mapping table
      // Get the current node
      node = GetNode(node_pid);
      if (IsDeleted(node)) {
        return;
      }
      
      uint32_t num_entries = 0;
      if (!is_inner) {
        if (node->node_type == Node::NodeType::Leaf) {
          num_entries = static_cast<LeafNode*>(node)->num_entries + 1;
        } else {
          num_entries = static_cast<DeltaNode*>(node)->num_entries + 1;
        }
      } else {
        if (node->node_type == Node::NodeType::Leaf) {
          num_entries = static_cast<InnerNode*>(node)->num_entries + 1;
        } else {
          num_entries = static_cast<DeltaNode*>(node)->num_entries + 1;
        }
      }

      if (num_entries < insert_branch_factor) return;

      Node* cas_node;
      if (!is_inner) {
        std::vector<std::pair<KeyType, ValueType>> kv_vec;
        CollapseLeafData(node, kv_vec);
        
        int split_idx = kv_vec.size() / 2;
        std::vector<std::pair<KeyType, ValueType>> kv_split_vec(
          std::make_move_iterator(kv_vec.begin() + split_idx),
          std::make_move_iterator(kv_vec.end()));
        split_key = std::get<0>(kv_vec[split_idx - 1]);
        KeyType low_key = std::get<0>(kv_split_vec[0]);
        KeyType high_key = std::get<0>(kv_split_vec[kv_split_vec.size() - 1]);
        pid_t left_link = node_pid;
        pid_t right_link = GetLeaf(node)->right_link;

        LeafNode *right_sibling = LeafNode::Create(low_key, high_key, right_link, left_link, kv_split_vec.size());
        int i = 0;
        for (auto it = std::make_move_iterator(kv_split_vec.begin()),
                     end = std::make_move_iterator(kv_split_vec.end()); it != end; ++it) {
          right_sibling->keys[i] = std::move(it->first);
          right_sibling->vals[i] = std::move(it->second);
          i++;
        }
        sibling_pid = pid_allocator_++;
        mapping_table_.Insert(sibling_pid, right_sibling);

        DeltaSplit* delta_split = new DeltaSplit();
        delta_split->split_key = split_key;
        delta_split->new_right = sibling_pid;
        delta_split->node_type = Node::NodeType::DeltaSplit;
        delta_split->num_entries = kv_vec.size() - kv_split_vec.size();
        LOG_DEBUG("Splitting leaf node with left node size %d right node size %lu", delta_split->num_entries, kv_split_vec.size());
        delta_split->next = node;
        cas_node = delta_split;
      } else {
        std::vector<std::pair<KeyType, pid_t>> kv_vec;
        CollapseInnerNodeData(node, kv_vec);
        
        int split_idx = kv_vec.size() / 2;
        std::vector<std::pair<KeyType, pid_t>> kv_split_vec(
          std::make_move_iterator(kv_vec.begin() + split_idx),
          std::make_move_iterator(kv_vec.end()));
        split_key = std::get<0>(kv_vec[split_idx - 1]);
        KeyType low_key = std::get<0>(kv_split_vec[0]);
        KeyType high_key = std::get<0>(kv_split_vec[kv_split_vec.size() - 1]);
        pid_t left_link = node_pid;
        pid_t right_link = GetInner(node)->right_link;

        InnerNode *right_sibling = InnerNode::Create(low_key, high_key, right_link, left_link, kv_split_vec.size());
        LOG_DEBUG("Splitting inner node with new size %lu", kv_split_vec.size());
        int i = 0;
        for (auto it = std::make_move_iterator(kv_split_vec.begin()),
                     end = std::make_move_iterator(kv_split_vec.end()); it != end; ++it) {
          right_sibling->keys[i] = std::move(it->first);
          right_sibling->children[i] = std::move(it->second);
          i++;
        }
        sibling_pid = pid_allocator_++;
        mapping_table_.Insert(sibling_pid, right_sibling);

        DeltaSplit* delta_split = new DeltaSplit();
        delta_split->split_key = split_key;
        delta_split->node_type = Node::NodeType::DeltaSplitInner;
        delta_split->new_right = sibling_pid;
        delta_split->num_entries = kv_vec.size() - kv_split_vec.size();
        delta_split->next = node;
        cas_node = delta_split;
      }
      bool cas_succ = mapping_table_.Cas(node_pid, node, cas_node);
      if (cas_succ) break;
      delete cas_node;
    } while (true);
    DeltaIndex* delta_index = new DeltaIndex();
    delta_index->node_type = Node::NodeType::DeltaIndex;
    delta_index->low_key = split_key;
    delta_index->new_child_pid = sibling_pid;
    LOG_DEBUG("CAS SUCCESSFUL! new delta index has new_child_pid %lu", delta_index->new_child_pid);
    if (parent_search_keys.empty()) {
      //This can't happen for inner splits
      delta_index->is_rightmost = true;
    } else {
      if (is_inner) {
        if (parent_search_keys.size() != 1) {
          parent_search_keys.pop();
          delta_index->high_key = parent_search_keys.top();
        }
      } else {
        delta_index->high_key = parent_search_keys.top();
      }
    }
    InstallDeltaIndex(node_pid, delta_index, is_inner);
  }

  void InstallDeltaIndex(pid_t orig_left, DeltaIndex* delta_index, bool is_inner)  {
    Node* node = nullptr;
    LOG_DEBUG("Installing delta index for is_inner: %d orig_left: %lu newright %lu", is_inner, orig_left, delta_index->new_child_pid);
    do {
      //orig_left was root
      pid_t parent; FindDataNodeResult result;
      result = FindDataNode(delta_index->low_key);
      parent = result.last_inner_node;
      if (is_inner) {
        if (result.traversal_path.size() == 1) {
          //Splitting root
          parent = kInvalidPid;
          LOG_DEBUG("Splitting orig_left %lu which was root", orig_left);
        } else {
          result.traversal_path.pop();
          parent = result.traversal_path.top();
          LOG_DEBUG("Splitting orig_left %lu which was not root whose parent is %lu", orig_left, parent);
        }
      }
      if (parent == kInvalidPid) {
        LOG_DEBUG("Installing delta index for root orig_left %lu", orig_left);
        //TODO what should the high key be here?
        InnerNode* inner = InnerNode::Create(delta_index->low_key, delta_index->high_key, kInvalidPid, kInvalidPid, 1);
        inner->keys[0] = delta_index->low_key;
        inner->children[0] = orig_left;
        inner->children[1] = delta_index->new_child_pid;
        pid_t new_root_pid = pid_allocator_++;
        mapping_table_.Insert(new_root_pid, inner);
        if (root_pid_.compare_exchange_weak(orig_left, new_root_pid)) {
          LOG_DEBUG("Installing new root successful! new root pid is now %lu with innernode at %p with left child %lu right child %lu", new_root_pid, inner, inner->children[0], inner->children[1]);
          break;
        } else {
          delete inner;
        }
      } else {
        node = GetNode(parent); 
        std::vector<std::pair<KeyType, pid_t>> output;
        CollapseInnerNodeData(node, output);
        pid_t left = orig_left; pid_t right = delta_index->new_child_pid;
        if (std::find_if(output.begin(), output.end(), 
              [left](std::pair<KeyType, pid_t> p) {
                return std::get<1>(p) == left;
              }) != output.end() &&
            std::find_if(output.begin(), output.end(), 
              [right](std::pair<KeyType, pid_t> p) {
                return std::get<1>(p) == right;
              }) != output.end() ) {
          delete delta_index;
          return;
        }
        //TODO: figure out if this needs to be prev_node.num_entries + 1
        delta_index->num_entries = output.size() + 1;
        delta_index->next = node;
        bool cas_succ = mapping_table_.Cas(parent, node, delta_index);
        if (cas_succ) break;
      }
    } while(true);
  }

  
   // Get the node with the given pid
  Node* GetNode(pid_t node_pid) const {
    return mapping_table_.Get(node_pid);
  }

  // Get the root node
  Node* GetRoot() const {
    return GetNode(root_pid_.load());
  }

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

  bool IsDeleted(const Node* node) const {
    return node->node_type == Node::NodeType::DeltaRemoveLeaf ||
           node->node_type == Node::NodeType::DeltaRemoveInner;
  }

  // Check if two keys are equal
  bool EqualKeys(KeyType k1, KeyType k2) const {
    return key_equals_(k1, k2);
  }

  // Check if two values are equal
  bool EqualValues(ValueType v1, ValueType v2) const {
    return value_comparator_.Compare(v1, v2);
  }

  // Delete the memory for this node's delta chain and base node
  void FreeNode(Node* node) {
    LOG_DEBUG("Freeing node (%p)", node);
    uint32_t chain_length = 0;
    Node* curr = node;
    while (curr->node_type != Node::NodeType::Inner &&
           curr->node_type != Node::NodeType::Leaf) {
      DeltaNode* delta = static_cast<DeltaNode*>(curr);
      curr = delta->next;
      delete delta;
      chain_length++;
    }
    auto node_type = curr->node_type;

    // Delete the base node
    delete curr;

    LOG_DEBUG("Freed node (%p) type %d with chain length of %u",
              node, node_type, chain_length);
  }

 private:
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
  uint32_t insert_branch_factor = 1;
  uint32_t chain_length_threshold = 10;

  std::atomic<uint64_t> num_failed_cas_;
  std::atomic<uint64_t> num_consolidations_;
};

}  // End index namespace
}  // End peloton namespace
