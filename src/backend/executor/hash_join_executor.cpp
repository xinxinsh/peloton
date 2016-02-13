//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// hash_join_executor.cpp
//
// Identification: src/backend/executor/hash_join_executor.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "backend/common/types.h"
#include "backend/common/logger.h"
#include "backend/executor/logical_tile_factory.h"
#include "backend/executor/hash_join_executor.h"
#include "backend/expression/abstract_expression.h"
#include "backend/expression/container_tuple.h"

namespace peloton {
namespace executor {

/**
 * @brief Constructor for hash join executor.
 * @param node Hash join node corresponding to this executor.
 */
HashJoinExecutor::HashJoinExecutor(const planner::AbstractPlan* node,
                                   ExecutorContext* executor_context)
    : AbstractJoinExecutor(node, executor_context) {}

bool HashJoinExecutor::DInit() {
  assert(children_.size() == 2);

  bool status = AbstractJoinExecutor::DInit();
  if (!status) return status;

  assert(children_[1]->GetRawNode()->GetPlanNodeType() == PLAN_NODE_TYPE_HASH);

  hash_executor_ = reinterpret_cast<HashExecutor*>(children_[1]);

  return true;
}

/**
 * @brief Creates logical tiles from the two input logical tiles after applying
 * join predicate.
 * @return true on success, false otherwise.
 */
bool HashJoinExecutor::DExecute() {
  LOG_INFO("Hash Join Executor");

  // Loop until we have non-empty result join logical tile or exit
  while (true) {
    // if (!buffered_output_tiles.empty()) {
    if (!result.empty()) {
      auto* output_tile = result.back();
      result.pop_back();
      SetOutput(output_tile);
      return true;
    }

    // Build outer join output when done
    if (left_child_done_ && right_child_done_) {
      return BuildOuterJoinOutput();
    }

    //===--------------------------------------------------------------------===//
    // Pick right and left tiles
    //===--------------------------------------------------------------------===//

    // Get all the logical tiles from RIGHT child
    if (!right_child_done_) {
      while (children_[1]->Execute()) {
        BufferRightTile(children_[1]->GetOutput());
      }
      right_child_done_ = true;
      LOG_INFO("Hash Join Executor: Got all %lu right tiles.",
               right_result_tiles_.size());
    }

    // Get next logical tile from LEFT child
    if (children_[0]->Execute()) {
      BufferLeftTile(children_[0]->GetOutput());
      LOG_INFO("Hash Join Executor: Got left tile %p.",
               left_result_tiles_.back().get());
    } else {
      // Left input is exhausted, loop around
      left_child_done_ = true;
      return BuildOuterJoinOutput();
    }
    if (right_result_tiles_.empty()) {
      /// No right children, a hash lookup would be empty. Continue ...
      continue;
    }

    //===--------------------------------------------------------------------===//
    // Build Join Tile
    //===--------------------------------------------------------------------===//

    LogicalTile* left_tile = left_result_tiles_.back().get();
    std::unordered_map<size_t,
                       std::unique_ptr<LogicalTile::PositionListsBuilder>>
        right_matches;

    // Get the hash table from the hash executor
    auto& hash_table = hash_executor_->GetHashTable();
    auto& hash_columns = hash_executor_->GetHashKeyIds();

    for (oid_t left_tid : *left_tile) {
      /// Create key and probe hash table
      HashExecutor::HashMapType::key_type key(left_tile, left_tid,
                                              &hash_columns);
      const auto& iter = hash_table.find(key);
      if (iter == hash_table.end()) {
        continue;
      }
      auto& matches = iter->second;
      for (auto& match : matches) {
        auto right_tile_index = match.first;
        auto* right_tile = right_result_tiles_[right_tile_index].get();
        auto right_tid = match.second;
        RecordMatchedRightRow(right_tile_index, right_tid);
        RecordMatchedLeftRow(left_result_tiles_.size() - 1, left_tid);

        const auto& pos_match_iter = right_matches.find(right_tile_index);
        if (pos_match_iter == right_matches.end()) {
          std::unique_ptr<LogicalTile::PositionListsBuilder> builder{
              new LogicalTile::PositionListsBuilder(left_tile, right_tile)};
          right_matches.insert(
              std::make_pair(right_tile_index, std::move(builder)));
        }
        right_matches[right_tile_index]->AddRow(left_tid, right_tid);
      }
    }

    // Create a new logical tile for every grouped match in matches
    for (auto& iter : right_matches) {
      auto output_tile = BuildOutputLogicalTile(
          left_tile, right_result_tiles_[iter.first].get());
      auto& pos_lists_builder = iter.second;
      output_tile->SetPositionListsAndVisibility(pos_lists_builder->Release());
      result.push_back(output_tile.release());
    }
  }
}

}  // namespace executor
}  // namespace peloton
