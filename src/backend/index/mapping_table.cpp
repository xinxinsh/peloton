//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// mapping_table.cpp
//
// Identification: src/backend/index/mapping_table.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "backend/index/mapping_table.h"

namespace peloton {
namespace index {

template <typename KeyType, typename ValueType, typename Hasher>
MappingTable<KeyType, ValueType, Hasher>::MappingTable(): MappingTable(1 << 20) {
  /* Nothing */
}

template <typename KeyType, typename ValueType, typename Hasher>
MappingTable<KeyType, ValueType, Hasher>::MappingTable(uint32_t initial_size) {
  table_ = new Entry[initial_size];
}

// Get the mapping for the provided key
template <typename KeyType, typename ValueType, typename Hasher>
ValueType MappingTable<KeyType, ValueType, Hasher>::Get(KeyType key) {
  auto* entry = EntryFor(key);
  return entry == nullptr ? nullptr : entry->val.load();
}

// CAS in the mapping for the provided key whose expected value is also provided
template <typename KeyType, typename ValueType, typename Hasher>
bool MappingTable<KeyType, ValueType, Hasher>::Cas(KeyType key, ValueType old_val, ValueType new_val) {
  auto* entry = EntryFor(key);
  if (entry == nullptr) {
    return false;
  }
  entry->val.compare_exchange_strong(old_val, new_val);
}

// Perform a blind set
template <typename KeyType, typename ValueType, typename Hasher>
bool MappingTable<KeyType, ValueType, Hasher>::Insert(KeyType key, ValueType val) {
  auto* curr_entry = EntryFor(key);
  if (curr_entry != nullptr) {
    return false;
  }
  uint64_t hash_value = HashFor(key);
  table_.push_back(Entry{hash_value, key, val});
}

}  // End index namespace
}  // End peloton namespace
