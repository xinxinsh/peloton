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

#include <cstdint>
#include <type_traits>

namespace peloton {
namespace index {

#if __GNUG__ && __GNUC__ < 5
#define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

template <typename KeyType, typename ValueType, typename Hasher>
class MappingTable {
 public:
  // Constructor
  MappingTable();
  MappingTable(uint32_t initial_size);

  // Get the mapping for the provided key
  ValueType get(KeyType key);

  // CAS in the mapping for the provided key whose expected value is also provided
  bool cas(KeyType key, ValueType old_val, ValueType new_val);

  // Perform a blind set
  bool insert(KeyType key, ValueType val);

 private:
  struct Entry {
    uint64_t hash;
    KeyType key;
    ValueType val;
  };

  // The mapping table. Note that the value must be copyable
  Entry* table_;
  //static_assert(IS_TRIVIALLY_COPYABLE(ValueType));

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
