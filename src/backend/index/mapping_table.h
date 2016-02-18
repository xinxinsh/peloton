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
  MappingTable();
  MappingTable(uint32_t initial_size);

  // Destructor
  ~MappingTable();

  // Get the mapping for the provided key
  ValueType Get(KeyType key);

  // CAS in the mapping for the provided key whose expected value is also provided
  bool Cas(KeyType key, ValueType old_val, ValueType new_val);

  // Perform a blind set
  bool Insert(KeyType key, ValueType val);

 private:
  // The structures we actually store in the mapping table
  struct Entry {
    uint64_t hash;
    KeyType key;
    std::atomic<ValueType> val;
  };

  // Get the entry for the provided key
  Entry* EntryFor(KeyType& key);

  // Hash a given key
  uint64_t HashFor(KeyType key) const { return hasher_(key); }

  // The mapping table
  Entry* table_;

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
