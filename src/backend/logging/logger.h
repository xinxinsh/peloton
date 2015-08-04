/*-------------------------------------------------------------------------
 *
 * logger.h
 * file description
 *
 * Copyright(c) 2015, CMU
 *
 * /peloton/src/backend/logging/logger.h
 *
 *-------------------------------------------------------------------------
 */

#pragma once

#include "backend/logging/logrecord.h"

#include <mutex>
#include <vector>

namespace peloton {
namespace logging {

//===--------------------------------------------------------------------===//
// Logger 
//===--------------------------------------------------------------------===//

// log queue
static std::vector<LogRecord> queue;

static std::mutex queue_mutex;

class Logger{

  public:

    Logger() {}

    static Logger& GetInstance();

    void logging_MainLoop(void);

    void Record(LogRecord queue);

    size_t GetBufferSize() const;

    void Flush();
 
};

}  // namespace logging
}  // namespace peloton
