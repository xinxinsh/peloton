//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// MappingTable.h
//
// Identification: src/backend/index/mapping_table.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

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

}  // End index namespace
}  // End peloton namespace
