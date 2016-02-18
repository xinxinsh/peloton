//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// bwtree.cpp
//
// Identification: src/backend/index/bwtree.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "backend/index/bwtree.h"

namespace peloton {
namespace index {

// Our map from NodeType => std::string for helpful error messages
// Templates make this guy fugly
template <typename KeyType, typename ValueType, class KeyComparator>
const std::map<
    typename BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType,
    std::string> BWTree<KeyType, ValueType,
                        KeyComparator>::kNodeTypeToString = {
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::Inner,
     "InnerNode"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::Leaf,
     "LeadNode"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaInsert,
     "Delta Insert"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaDelete,
     "Delta Delete"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaMerge,
     "Delta Merge"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaMergeInner,
     "Delta Merge Inner"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaSplit,
     "Delta Split"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaSplitInner,
     "Delta Split Inner"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaIndex,
     "Delta Index"},
    {BWTree<KeyType, ValueType,
            KeyComparator>::Node::NodeType::DeltaDeleteIndex,
     "Delta Delete Index"},
    {BWTree<KeyType, ValueType, KeyComparator>::Node::NodeType::DeltaRemoveLeaf,
     "Delta Remove Leaf Node"},
    {BWTree<KeyType, ValueType,
            KeyComparator>::Node::NodeType::DeltaRemoveInner,
     "Delta Remove Inner Node"}};

}  // End index namespace
}  // End peloton namespace
