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
      LeafNode* inner = new InnerNode();
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
    // TODO: fix this by refactoring or sth
    // Zero for DeltaSplitInner
    // num_entries is sizeo f the original branch (ie left)
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
    bool found = false;
    // The slot to find the value
    uint32_t slot_idx = 0;
    // The PID of the leaf node that contains the value
    pid_t node_pid = kInvalidPid;
    // The head (root) of the delta chain (if one exists)
    Node* head = nullptr;
    // The data was either found in the leaf node or a delta node.  Only one of
    // the following two pointers should be non-null if found is true.
    DeltaNode* delta_node = nullptr;
    LeafNode* leaf_node = nullptr;
    // The path the search took
    std::stack<pid_t> traversal_path;
    // The PID of the first node we find while traversal that needs consolidation
    pid_t needs_consolidation = kInvalidPid;
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
  }

  // Insertion
  bool Insert(KeyType key, ValueType value) {
    FindDataNodeResult result = FindDataNode(key);

    Node* prev_root = result.head;
    uint32_t num_entries = 0;
    if (result.leaf_node) {
      num_entries = result.leaf_node->num_entries + 1;
    } else {
      DeltaNode* delta = static_cast<DeltaNode*>(result.head);
      num_entries = delta->num_entries + 1;
    }

    // Collect values
    std::vector<std::pair<KeyType, ValueType>> vals;
    CollapseLeafData(result.head, vals);

    // Check if this is a duplicate key-value pair by binary searching
    // the values from the leaf we intend to insert into
    auto range = std::equal_range(vals.begin(), vals.end(), key,
                                  KeyOnlyComparator{key_comparator_});
    for (auto iter = range.first, end = range.second; iter != end; ++iter) {
      std::pair<KeyType, ValueType>& pair = *iter;
      if (value_comparator_.Compare(pair.second, value)) {
        LOG_DEBUG("Attempted to insert duplicate key-value pair");
        return false;
      }
    }

    // TODO If remove, goto left sibling and insert there in multithreading case
    DeltaInsert* delta_insert = new DeltaInsert();
    delta_insert->key = key;
    delta_insert->value = value;
    // TODO: make this class into c++ initializer list
    delta_insert->node_type = Node::NodeType::DeltaInsert;
    delta_insert->next = prev_root;

    pid_t root_pid = result.node_pid;
    while (!mapping_table_.Cas(root_pid, prev_root, delta_insert)) {
      // TODO: need to retraverse in multithreading
      prev_root = mapping_table_.Get(root_pid);
      delta_insert->next = prev_root;
      num_failed_cas_++;
    }

    LOG_DEBUG("Inserted new index entry");

    if (num_entries > insert_branch_factor) {
      // Split(result.leaf_node, node_pid, delta_insert);
    }
    if (result.needs_consolidation != kInvalidPid) {
      ConsolidateNode(result.needs_consolidation);
    }

    return true;
  }

  bool Delete(KeyType key, const ValueType value) {
    FindDataNodeResult result = FindDataNode(key);

    assert(IsLeaf(result.head));

    Node* prev_root = result.head;
    uint32_t num_entries = 0;
    if (result.leaf_node) {
      num_entries = result.leaf_node->num_entries - 1;
    } else {
      DeltaNode* delta = static_cast<DeltaNode*>(result.head);
      num_entries = delta->num_entries - 1;
    }

    // Get the list of key-value pairs in the leaf
    std::vector<std::pair<KeyType, ValueType>> vals;
    CollapseLeafData(result.head, vals);

    // Check if this key-value pair exists in the tree by binary searching
    // the values from the leaf where the data exists
    bool matched = false;
    auto range = std::equal_range(vals.begin(), vals.end(), key,
                                  KeyOnlyComparator{key_comparator_});
    for (auto iter = range.first, end = range.second; iter != end; ++iter) {
      std::pair<KeyType, ValueType>& pair = *iter;
      if (value_comparator_.Compare(pair.second, value)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
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
    LOG_DEBUG("Inserted delta delete to node %lu", result.node_pid);
    return true;
  }

  // Return an iterator that points to the element that is associated with the
  // provided key, or an iterator that is equal to what is returned by end()
  Iterator Search(const KeyType key) {
    // Find the data node where the key may be
    FindDataNodeResult result = FindDataNode(key);

    // Collapse the chain+delta into a vector of values
    assert(IsLeaf(result.head));
    std::vector<std::pair<KeyType, ValueType>> vals;
    CollapseLeafData(result.head, vals);

    auto found_pos = std::lower_bound(vals.begin(), vals.end(), key,
                                      KeyOnlyComparator{key_comparator_});
    if (found_pos == vals.end()) {
      return end();
    }

    uint32_t slot = found_pos - vals.begin();

    LOG_DEBUG("Found key in leaf slot %d", slot);

    return Iterator{*this, slot, result.node_pid, result.head, std::move(vals)};
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
  FindDataNodeResult FindDataNode(const KeyType key) const {
    // The result
    FindDataNodeResult result;

    // The path we take during the traversal/search
    std::stack<pid_t> traversal;
    // The PID of the node we're currently probing
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
        if (IsDeleted(curr_node)) {
          curr = traversal.top();
          traversal.pop();
        } else {
          break;
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
          auto iter =
              std::lower_bound(inner->keys, inner->keys + inner->num_entries,
                               key, key_comparator_);
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
          // This node has been deleted (presumably from a merge to another
          // node). Go back up the traversal path and try again
          return kInvalidPid;
        }
        default:
          // Anything else should be impossible for inner nodes
          //const std::string type = kNodeTypeToString[curr_node->node_type];
          LOG_DEBUG("Hit node on inner node traversal.  This is impossible!");
          assert(false);
      }
    }
    // Should never happen
    assert(false);
  }

  // Find the given key in the provided leaf node.  If found, assign the value
  // reference to the value from the leaf and return true. Otherwise, return
  // false if not found in the leaf.
  void FindInLeafNode(const Node* node, const KeyType key,
                      FindDataNodeResult& result) const {
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
          if (key_equals_(key, insert->key)) {
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
          if (key_equals_(key, del->key)) {
            // The key/value was deleted
            result.head = (Node*) node;
            result.found = false;
            result.slot_idx = 0;
            result.delta_node = del;
            result.leaf_node = nullptr;
            return;
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
            curr = GetNode(merge->new_right);
          }
          break;
        }
        case Node::NodeType::DeltaSplit: {
          DeltaSplit* split = static_cast<DeltaSplit*>(curr);
          if (key_comparator_(key, split->split_key)) {
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
    }
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
          CollapseLeafData(GetNode(merge->new_right), output);
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

    LOG_DEBUG(
        "CollapseLeafData: Found %lu inserted, %lu deleted, chain length %d",
        inserted.size(), deleted.size(), chain_length + 1);

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

    LOG_DEBUG("Collapsed size before insertions and deletes: %lu",
              output.size());

    // Add inserted key-value pairs from the delta chain
    for (uint32_t i = 0; i < inserted.size(); i++) {
      auto pos =
          std::lower_bound(output.begin(), output.end(), inserted[i], cmp);
      output.insert(pos, inserted[i]);
    }

    LOG_DEBUG("Collapsed size after insertions: %lu", output.size());

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

  void ConsolidateInnerNode(__attribute((unused)) pid_t node_pid) {
  }

  void ConsolidateLeafNode(pid_t node_pid) {
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
        return;
      }

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
  }

  // For Insert
  // template <typename KeyType, typename ValueType, class KeyComparator>
  KVMultiset& getKVsLeaf(Node *node, KVMultiset& deltaKVs, std::set<KeyType>& deleted) {
    for (int i = 0; i < node->num_entries; i++) {
      auto it = deleted.find(node->keys[i]);
      if (it == deleted.end()) {
        deltaKVs->insert(std::make_tuple(node->keys[i], node->vals[i]));
      }
    }
    return deltaKVs;
  }

  KVMultiset& getKVsDeltaAndLeaf(DeltaNode *node) {
    Node* cur = node->next;
    // TODO: unique ptr?
    KVMultiset* kvs = new std::multiset<std::pair<KeyType, ValueType>>();
    std::set<KeyType> deleted = new std::set<KeyType>();
    while (cur) {
      switch (cur->node_type){
        case Node::NodeType::DeltaInsert: {
          auto it = deleted.find(cur->key);
          if (it == deleted.end()) {
            auto t = std::make_tuple(cur->key, cur->value);
            kvs.insert(t);
          }
          break;
        }
        case Node::NodeType::DeltaDelete: {
          deleted.insert(cur->key);
          break;
        }
        case Node::NodeType::DeltaMerge: {
          // TODO: Watch out for stale references in multithreading mode.
          KVMultiset& right_set = getKVsAndCountDelta(mapping_table_.Get(cur->new_right));
          kvs.insert(right_set.begin(), right_set.end());
          break;
        }
        case Node::NodeType::DeltaSplit:
          //TODO: this might not be the case in multithreading.
          // Keep traversing down the current chain.
          break;
        case Node::NodeType::Leaf: {
          return getKVsAndCountLeaf(cur, kvs, deleted);
        }
        default: {
          // TODO: With single threading, nothing in the chain can be DeltaRemoveLeaf.
          // In multithreading, there might be some weird cases.
          LOG_DEBUG("hit node %s on insert.  this is impossible!",
                    std::to_string(cur->node_type).c_str());
          assert(false);
        }
      }
      cur = cur->next;
    }
  }

  void Split(LeafNode* leaf, pid_t leaf_pid, DeltaNode* node) {
    //TODO: Handle duplicate keys
    KVMultiset& KVs = getKVsDeltaAndLeaf(node);
    std::vector<std::pair<KeyType, ValueType>> kv_vec(KVs.begin(), KVs.end());
    int split_idx = kv_vec.size() / 2;
    std::vector<std::pair<KeyType, ValueType>> kv_split_vec(
        std::make_move_iterator(kv_vec.begin() + split_idx),
        std::make_move_iterator(kv_vec.end()));
    std::vector<KeyType>* split_vec_keys = new std::vector<KeyType>(kv_split_vec.size());
    std::vector<ValueType>* split_vec_values = new std::vector<ValueType>(kv_split_vec.size());
    for (auto it = std::make_move_iterator(kv_split_vec.begin()),
                 end = std::make_move_iterator(kv_split_vec.end()); it != end; ++it) {
      split_vec_keys.push_back(std::move(it->first));
      split_vec_values.push_back(std::move(it->second));
    }
    KeyType split_key = std::get<KeyType>(kv_vec[split_idx - 1]);
    KeyType low_key = std::get<KeyType>(kv_split_vec[0]);
    KeyType high_key = std::get<KeyType>(kv_split_vec[kv_split_vec.size() - 1]);
    pid_t left_link = leaf_pid;
    pid_t right_link = node->right_link;

    //TODO: use initializer lists
    LeafNode *right_sibling = new LeafNode();

    right_sibling->low_key = low_key;
    right_sibling->high_key = high_key;
    right_sibling->right_link = right_link;
    right_sibling->left_link = left_link;
    right_sibling->num_entries = kv_split_vec.size();
    right_sibling->keys = &split_vec_keys[0];
    right_sibling->vals = &split_vec_values[0];

    right_sibling->next = node;
    right_sibling->node_type = Node::NodeType::Leaf;
    pid_t sibling_pid = pid_allocator_++;
    mapping_table_.insert(sibling_pid, right_sibling);
    //TODO: what happens wit multithreading?
    leaf->right_link = sibling_pid;

    DeltaSplit* delta_split = new DeltaSplit();
    delta_split->split_key = split_key;
    delta_split->new_right = sibling_pid;
    delta_split->num_entries = kv_vec.size() - kv_split_vec.size();

    //TODO: need to retraverse in multithreading;
    mapping_table_.Cas(leaf_pid, node, delta_split);

    // Install index
    DeltaSplit* split_index = new DeltaSplit();
    split_index->node_type = Node::NodeType::DeltaSplitInner;
    split_index->split_key = split_key;
    split_index->new_right = sibling_pid;
    // TODO: fix this for multithreading
    split_index->next = node;

    //TODO: need to retraverse in multithreading;
    mapping_table_.Cas(leaf_pid, delta_split, split_index);
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
  // TODO: just a randomly chosen number now...
  int delete_branch_factor = 100;

  int insert_branch_factor = 500;

  std::atomic<uint64_t> num_failed_cas_;
  std::atomic<uint64_t> num_consolidations_;
};

}  // End index namespace
}  // End peloton namespace
