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
#include <thread>

namespace peloton {
namespace index {

//===--------------------------------------------------------------------===//
// The epoch manager tracks epochs in the system and arranges for memory
// reclamation at safe points. There is no global GC, but rather, each thread
// uses information about it's own epoch, the current epoch and lowest
// epoch that a thread is current in to determine whether memory it has
// marked as deleted can actually be physically deleted.
//===--------------------------------------------------------------------===//
class EpochManager {
 private:
  typedef uint64_t epoch_t;
  static const uint64_t kEpochTimer = 40;

  // Represents a group of items belonging to the same epoch that can be
  // deleted together
  static const uint32_t kGroupSize = 32;
  struct DeletionGroup {
    std::array<char*, kGroupSize> items;
    epoch_t epoch;
    uint32_t num_items = 0;
    DeletionGroup* next_group = nullptr;
  };

  struct DeletionList {
    DeletionGroup* head = nullptr;
    DeletionGroup* tail = nullptr;
    DeletionGroup* free_to_use_groups = nullptr;

    ~DeletionList() {
      while (head != nullptr && head->next_group != nullptr) {
        DeletionGroup* tmp = head;
        head = head->next_group;
        delete tmp;
      }
      while (free_to_use_groups != nullptr &&
             free_to_use_groups->next_group != nullptr) {
        DeletionGroup* tmp = free_to_use_groups;
        free_to_use_groups = free_to_use_groups->next_group;
        delete tmp;
      }
    }

    void MarkDeleted(char* ptr, epoch_t epoch) {
      DeletionGroup* to_add = nullptr;

      if (head == nullptr) {
        assert(tail == nullptr);
        to_add = new DeletionGroup();
        to_add->epoch = epoch;
        to_add->num_items = 0;
        to_add->next_group = nullptr;
        tail = head = to_add;
      } else if (tail->epoch != epoch || tail->num_items >= kGroupSize) {
        if (free_to_use_groups != nullptr) {
          to_add = free_to_use_groups;
          free_to_use_groups = free_to_use_groups->next_group;
        } else {
          to_add = new DeletionGroup();
        }
        to_add->epoch = epoch;
        to_add->num_items = 0;
        to_add->next_group = nullptr;
        tail->next_group = to_add;
        tail = to_add;
      }

      // The tail group has the same epoch and has room
      to_add->items[to_add->num_items++] = ptr;
    }
  };

  // Captures the state information for every thread that participates in
  // epoch-based garbage collection.
  // Note: None of these members need to be protected through a mutex
  //       because we guarantee that all mutations are performed by
  //       a single thread (the thread whose state it represents)
  struct ThreadState {
    bool initialized;
    epoch_t local_epoch;
    DeletionList deletion_list;
    uint32_t added = 0;
    uint32_t deleted = 0;
  };

 public:
  // Constructor
  EpochManager()
      : global_epoch_(0),
        low_epoch_(0),
        stop_(false),
        epoch_mover_() {
    /* Nothing */
  }

  // Destructor
  ~EpochManager() {
    stop_.store(true);
    epoch_mover_.join();
  }

  void Init() {
    epoch_mover_ = std::thread{[=]() {
      LOG_DEBUG("Starting epoch incrementer thread");
      std::chrono::milliseconds sleep_duration{kEpochTimer};
      while (!stop_.load()) {
        std::this_thread::sleep_for(sleep_duration);
        LOG_DEBUG("Epoch incrementer woke up, incrementing epoch to %lu",
                  ++global_epoch_);
      }
    }};
  }

  // Called by threads that want to enter the current epoch, indicating that
  // it will need all resources that are visible in this epoch
  void EnterEpoch() {
    auto* thread_state = GetThreadState();
    if (!thread_state->initialized) {
      // do init
      thread_state->initialized = true;
    }
    thread_state->local_epoch = GetCurrentEpoch();
  }

  // Called by a thread that wishes to mark the provided data ptr as deleted in
  // this epoch.  The data will not be deleted immediately, but only when it
  // it safe to do so, i.e., when all active threads have quiesced and exited
  // the epoch.
  void MarkDeleted(char* ptr) {
    auto* thread_state = GetThreadState();
    assert(thread_state->initialized);
    thread_state->deletion_list.MarkDeleted(ptr, thread_state->local_epoch);
  }

  // The thread wants to exit the epoch, indicating it no longer needs
  // protected access to the data
  void ExitEpoch() {
    auto* thread_state = GetThreadState();
    assert(thread_state->initialized);
  }

  // Get the current epoch
  epoch_t GetCurrentEpoch() const {
    return global_epoch_.load();
  }

 private:
  // Get the epoch thread state for the currently executing thread
  ThreadState* GetThreadState() {
    static thread_local ThreadState thread_state;
    return &thread_state;
  }

  // The global epoch number
  std::atomic<epoch_t> global_epoch_;
  // The current minimum epoch that any transaction is in
  std::atomic<epoch_t> low_epoch_;
  // The thread that increments the epoch
  std::atomic<bool> stop_;
  std::thread epoch_mover_;
};

//===--------------------------------------------------------------------===//
// A handy scoped guard that callers can use at the beginning of pieces of
// code they need to execute within an epoch(s)
//===--------------------------------------------------------------------===//
class EpochGuard {
 public:
  // Constructor (enters the current epoch)
  EpochGuard(EpochManager& em): em_(em) {
    em_.EnterEpoch();
  }
  // Desctructor (exits the epoch)
  ~EpochGuard() {
    em_.ExitEpoch();
  }
 private:
  // The epoch manager
  EpochManager& em_;
};

}  // End index namespace
}  // End peloton namespace
