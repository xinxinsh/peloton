//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// btree_index.h
//
// Identification: src/backend/index/btree_index.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>
#include <string>
#include <map>

#include "backend/catalog/manager.h"
#include "backend/common/platform.h"
#include "backend/common/types.h"
#include "backend/index/index.h"

#include "backend/index/bwtree.h"

namespace peloton {
namespace index {

class ItemPointerComparator {
 public:
  bool Compare(const ItemPointer x, const ItemPointer y) const {
    return (x.block == y.block) && (x.offset == y.offset);
  }
};

/**
 * BW tree-based index implementation.
 *
 * @see Index
 */
template <typename KeyType, typename ValueType, class KeyComparator,
          class KeyEqualityChecker>
class BWTreeIndex : public Index {
  friend class IndexFactory;

  typedef BWTree<KeyType, ValueType, KeyComparator, ItemPointerComparator,
                 KeyEqualityChecker> MapType;

 public:
  BWTreeIndex(IndexMetadata *metadata);

  ~BWTreeIndex();

  bool InsertEntry(const storage::Tuple *key, const ItemPointer location);

  bool DeleteEntry(const storage::Tuple *key, const ItemPointer location);

  std::vector<ItemPointer> Scan(const std::vector<Value> &values,
                                const std::vector<oid_t> &key_column_ids,
                                const std::vector<ExpressionType> &expr_types,
                                const ScanDirectionType &scan_direction);

  std::vector<ItemPointer> ScanAllKeys();

  std::vector<ItemPointer> ScanKey(const storage::Tuple *key);

  std::string GetTypeName() const;

  bool Cleanup() { return true; }

  size_t GetMemoryFootprint() { return container.GetTreeSize(); }

 protected:
  // container
  MapType container;

  // equality checker and comparator
  KeyEqualityChecker equals;
  KeyComparator comparator;

  // synch helper
  RWLock index_lock;
};

}  // End index namespace
}  // End peloton namespace
