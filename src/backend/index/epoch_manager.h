//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// epoch_manager.h
//
// Identification: src/backend/index/epoch_manager.h
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#pragma once

#include "backend/common/logger.h"

#include <array>
#include <atomic>
#include <cassert>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace peloton {
namespace index {

//===--------------------------------------------------------------------===//
// The epoch manager tracks epochs in the system and arranges for memory
// reclamation at safe points. There is no global GC, but rather, each thread
// uses information about it's own epoch, the current epoch and lowest
// epoch that a thread is current in to determine whether memory it has
// marked as deleted can actually be physically deleted.
//===--------------------------------------------------------------------===//
template <class Freeable>
class EpochManager {
 private:
  // The type of the epoch
  typedef uint64_t epoch_t;
  // The frequency the global epoch is updated
  static const uint64_t kEpochTimer = 40;

  // Represents a group of items belonging to the same epoch that can be
  // deleted together
  static const uint32_t kGroupSize = 32;
  struct DeletionGroup {
    std::array<Freeable, kGroupSize> items;
    epoch_t epoch;
    uint32_t num_items = 0;
    DeletionGroup* next_group = nullptr;
  };

  struct DeletionList {
    DeletionGroup* head = nullptr;
    DeletionGroup* tail = nullptr;
    DeletionGroup* free_to_use_groups = nullptr;

    ~DeletionList() {
      LOG_DEBUG("Freeing all memory in all deletion groups during shutdown");

      // Free all the stuff that was marked deleted, but was potentially in use
      // by another thread.  We can do this because we're shutting down.
      Free(std::numeric_limits<epoch_t>::max());

      // Reclaim memory we created for deletion groups
      while (head != nullptr) {
        DeletionGroup* tmp = head;
        head = head->next_group;
        delete tmp;
      }
      while (free_to_use_groups != nullptr) {
        DeletionGroup* tmp = free_to_use_groups;
        free_to_use_groups = free_to_use_groups->next_group;
        delete tmp;
      }
    }

    // Get a deletion group with sufficient space for items deleted in
    // the given eopch
    // TODO: Does it really have to be the same epoch?
    DeletionGroup* GetAvailableGroup(epoch_t epoch) {
      if (tail != nullptr && tail->epoch == epoch &&
          tail->num_items < kGroupSize - 1) {
        return tail;
      }

      // Either the tail is null, or has a different epoch or has no room
      // for new entries.  In any case, we need a new fresh group

      DeletionGroup* group = nullptr;
      if (free_to_use_groups != nullptr) {
        group = free_to_use_groups;
        free_to_use_groups = free_to_use_groups->next_group;
      } else {
        group = new DeletionGroup();
      }
      group->epoch = epoch;
      group->num_items = 0;
      group->next_group = nullptr;

      if (head == nullptr) {
        assert(tail == nullptr);
        head = tail = group;
      } else {
        assert(tail != nullptr);
        tail->next_group = group;
        tail = group;
      }
      return group;
    }

    void MarkDeleted(Freeable freeable, epoch_t epoch) {
      DeletionGroup* to_add = GetAvailableGroup(epoch);
      assert(to_add != nullptr);
      assert(to_add->epoch == epoch);
      assert(to_add->num_items < kGroupSize);
      to_add->items[to_add->num_items++] = freeable;
    }

    // Free all deleted data before the given epoch
    void Free(epoch_t epoch) {
      uint32_t freed = 0;
      while (head != nullptr && head->epoch < epoch) {
        DeletionGroup* next = head->next_group;
        // All items in the head deletion group can be freed
        for (uint32_t i = 0; i < head->num_items; i++, freed++) {
          head->items[i].Free();
        }
        // Add the deletion group to the free_to_use list
        head->next_group = free_to_use_groups;
        free_to_use_groups = head;
        head = next;
      }
      if (freed > 0) {
        LOG_DEBUG("Freed %u objects before epoch %lu", freed, epoch);
      }
    }
  };

  // Captures the state information for every thread that participates in
  // epoch-based garbage collection.
  // Note: None of these members need to be protected through a mutex
  //       because we guarantee that all mutations are performed by
  //       a single thread (the thread whose state it represents)
  struct ThreadState {
    bool initialized = false;
    volatile epoch_t local_epoch = 0;
    DeletionList deletion_list;
    uint32_t added = 0;
    uint32_t deleted = 0;
  };

  // Get the epoch thread state for the currently executing thread
  ThreadState* GetThreadState() {
    static thread_local ThreadState thread_state;
    return &thread_state;
  }

 public:
  // Constructor
  EpochManager()
      : global_epoch_(0), lowest_epoch_(0), stop_(false), epoch_mover_() {
    /* Nothing */
  }

  // Destructor
  ~EpochManager() {
    stop_.store(true);
    epoch_mover_.join();
  }

  void Init() {
    // Start the epoch ticker thread
    epoch_mover_ = std::thread{&EpochManager::EpochTicker, this};
  }

  void EpochTicker() {
    LOG_DEBUG("Starting epoch ticker thread");
    std::chrono::milliseconds sleep_duration{kEpochTimer};
    while (!stop_.load()) {
      // Sleep
      std::this_thread::sleep_for(sleep_duration);
      // Increment global epoch
      global_epoch_++;
      uint32_t num_states = 0;
      {
        // Find the lowest epoch number among active threads. Data with
        // epoch < lowest can be deleted
        std::lock_guard<std::mutex> lock{states_mutex_};
        num_states = thread_states_.size();
        epoch_t lowest = global_epoch_.load();
        for (const auto& iter : thread_states_) {
          ThreadState* state = iter.second;
          if (state->local_epoch < lowest) {
            lowest = state->local_epoch;
          }
        }
        lowest_epoch_.store(lowest);
      }
      LOG_DEBUG("Global epoch: %lu, lowest epoch: %lu, # states: %u",
                global_epoch_.load(), lowest_epoch_.load(), num_states);
      assert(lowest_epoch_.load() <= global_epoch_.load());
    }
    LOG_DEBUG("Shutting down epoch ticker thread");
  }

  // Called by threads that want to enter the current epoch, indicating that
  // it will need all resources that are visible in this epoch
  void EnterEpoch() {
    auto* self_state = GetThreadState();
    if (!self_state->initialized) {
      // do init
      self_state->initialized = true;
      std::lock_guard<std::mutex> lock{states_mutex_};
      thread_states_[std::this_thread::get_id()] = self_state;
    }
    self_state->local_epoch = GetCurrentEpoch();
  }

  // Called by a thread that wishes to mark the provided freeable entity as
  // deleted in this epoch. The data will not be deleted immediately, but only
  // when it it safe to do so, i.e., when all active threads have quiesced and
  // exited the epoch.
  //
  // Note: The Freeable type must have a Free(...) function that can be called
  //       to physically free the memory
  void MarkDeleted(Freeable freeable) {
    auto* self_state = GetThreadState();
    assert(self_state->initialized);

    // Get the deletion list for the thread and mark the ptr as deleted by
    // this thread in the thread's local epoch
    auto& deletion_list = self_state->deletion_list;
    deletion_list.MarkDeleted(freeable, self_state->local_epoch);
  }

  // The thread wants to exit the epoch, indicating it no longer needs
  // protected access to the data
  void ExitEpoch() {
    auto* self_state = GetThreadState();
    assert(self_state->initialized);
    auto& deletion_list = self_state->deletion_list;
    deletion_list.Free(lowest_epoch_.load());
  }

  // Get the current epoch
  epoch_t GetCurrentEpoch() const {
    return global_epoch_.load();
  }

 private:

  // The global epoch number
  std::atomic<epoch_t> global_epoch_;
  // The current minimum epoch that any transaction is in
  std::atomic<epoch_t> lowest_epoch_;
  // The thread that increments the epoch
  std::atomic<bool> stop_;
  std::thread epoch_mover_;
  // The map from threads to their states, and mutex that protects it
  std::mutex states_mutex_;
  std::unordered_map<std::thread::id, ThreadState*> thread_states_;
};

//===--------------------------------------------------------------------===//
// A handy scoped guard that callers can use at the beginning of pieces of
// code they need to execute within an epoch(s)
//===--------------------------------------------------------------------===//
template <class Type>
class EpochGuard {
 public:
  // Constructor (enters the current epoch)
  EpochGuard(EpochManager<Type>& em): em_(em) {
    em_.EnterEpoch();
  }
  // Desctructor (exits the epoch)
  ~EpochGuard() {
    em_.ExitEpoch();
  }
 private:
  // The epoch manager
  EpochManager<Type>& em_;
};

}  // End index namespace
}  // End peloton namespace
