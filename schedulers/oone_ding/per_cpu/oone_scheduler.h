// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef GHOST_SCHEDULERS_Oone_Oone_SCHEDULER_H
#define GHOST_SCHEDULERS_Oone_Oone_SCHEDULER_H

#include <deque>
#include <memory>

#include "lib/agent.h"
#include "lib/scheduler.h"

namespace ghost {

enum class OoneTaskState {
  kBlocked,   // not on runqueue.
  kRunnable,  // transitory state:
              // 1. kBlocked->kRunnable->kQueued
              // 2. kQueued->kRunnable->kOnCpu
  kQueued,    // on runqueue.
  kOnCpu,     // running on cpu.
};

// For CHECK and friends.
std::ostream& operator<<(std::ostream& os, const OoneTaskState& state);

struct OoneTask : public Task<> {
  explicit OoneTask(Gtid Oone_task_gtid, ghost_sw_info sw_info)
      : Task<>(Oone_task_gtid, sw_info) {}
  ~OoneTask() override {}

  inline bool blocked() const { return run_state == OoneTaskState::kBlocked; }
  inline bool queued() const { return run_state == OoneTaskState::kQueued; }
  inline bool oncpu() const { return run_state == OoneTaskState::kOnCpu; }

  // N.B. _runnable() is a transitory state typically used during runqueue
  // manipulation. It is not expected to be used from task msg callbacks.
  //
  // If you are reading this then you probably want to take a closer look
  // at queued() instead.
  inline bool _runnable() const {
    return run_state == OoneTaskState::kRunnable;
  }

  OoneTaskState run_state = OoneTaskState::kBlocked;
  int cpu = -1;

  // Whether the last execution was preempted or not.
  bool preempted = false;

  // A task's priority is boosted on a kernel preemption or a !deferrable
  // wakeup - basically when it may be holding locks or other resources
  // that prevent other tasks from making progress.
  bool prio_boost = false;
};

class OoneRq {
 public:
  OoneRq() = default;
  OoneRq(const OoneRq&) = delete;
  OoneRq& operator=(OoneRq&) = delete;

  OoneTask* Dequeue();
  void Enqueue(OoneTask* task);

  // Erase 'task' from the runqueue.
  //
  // Caller must ensure that 'task' is on the runqueue in the first place
  // (e.g. via task->queued()).
  void Erase(OoneTask* task);

  size_t Size() const {
    absl::MutexLock lock(&mu_);
    return rq_.size();
  }

  bool Empty() const { return Size() == 0; }

 private:
  mutable absl::Mutex mu_;
  std::deque<OoneTask*> rq_ ABSL_GUARDED_BY(mu_);
};

class OoneScheduler : public BasicDispatchScheduler<OoneTask> {
 public:
  explicit OoneScheduler(Enclave* enclave, CpuList cpulist,
                         std::shared_ptr<TaskAllocator<OoneTask>> allocator);
  ~OoneScheduler() final {}

  void Schedule(const Cpu& cpu, const StatusWord& sw);

  void EnclaveReady() final;
  Channel& GetDefaultChannel() final { return *default_channel_; };

  bool Empty(const Cpu& cpu) {
    CpuState* cs = cpu_state(cpu);
    return cs->run_queue.Empty();
  }

  void DumpState(const Cpu& cpu, int flags) final;
  std::atomic<bool> debug_runqueue_ = false;

  int CountAllTasks() {
    int num_tasks = 0;
    allocator()->ForEachTask([&num_tasks](Gtid gtid, const OoneTask* task) {
      ++num_tasks;
      return true;
    });
    return num_tasks;
  }

  static constexpr int kDebugRunqueue = 1;
  static constexpr int kCountAllTasks = 2;

 protected:
  void TaskNew(OoneTask* task, const Message& msg) final;
  void TaskRunnable(OoneTask* task, const Message& msg) final;
  void TaskDeparted(OoneTask* task, const Message& msg) final;
  void TaskDead(OoneTask* task, const Message& msg) final;
  void TaskYield(OoneTask* task, const Message& msg) final;
  void TaskBlocked(OoneTask* task, const Message& msg) final;
  void TaskPreempted(OoneTask* task, const Message& msg) final;
  void TaskSwitchto(OoneTask* task, const Message& msg) final;

 private:
  void OoneSchedule(const Cpu& cpu, BarrierToken agent_barrier,
                    bool prio_boosted);
  void TaskOffCpu(OoneTask* task, bool blocked, bool from_switchto);
  void TaskOnCpu(OoneTask* task, Cpu cpu);
  void Migrate(OoneTask* task, Cpu cpu, BarrierToken seqnum);
  Cpu AssignCpu(OoneTask* task);
  void DumpAllTasks();

  struct CpuState {
    OoneTask* current = nullptr;
    std::unique_ptr<Channel> channel = nullptr;
    OoneRq run_queue;
  } ABSL_CACHELINE_ALIGNED;

  inline CpuState* cpu_state(const Cpu& cpu) { return &cpu_states_[cpu.id()]; }

  inline CpuState* cpu_state_of(const OoneTask* task) {
    CHECK_GE(task->cpu, 0);
    CHECK_LT(task->cpu, MAX_CPUS);
    return &cpu_states_[task->cpu];
  }

  CpuState cpu_states_[MAX_CPUS];
  Channel* default_channel_ = nullptr;
};

std::unique_ptr<OoneScheduler> MultiThreadedOoneScheduler(Enclave* enclave,
                                                          CpuList cpulist);
class OoneAgent : public LocalAgent {
 public:
  OoneAgent(Enclave* enclave, Cpu cpu, OoneScheduler* scheduler)
      : LocalAgent(enclave, cpu), scheduler_(scheduler) {}

  void AgentThread() override;
  Scheduler* AgentScheduler() const override { return scheduler_; }

 private:
  OoneScheduler* scheduler_;
};

template <class EnclaveType>
class FullOoneAgent : public FullAgent<EnclaveType> {
 public:
  explicit FullOoneAgent(AgentConfig config) : FullAgent<EnclaveType>(config) {
    scheduler_ =
        MultiThreadedOoneScheduler(&this->enclave_, *this->enclave_.cpus());
    this->StartAgentTasks();
    this->enclave_.Ready();
  }

  ~FullOoneAgent() override {
    this->TerminateAgentTasks();
  }

  std::unique_ptr<Agent> MakeAgent(const Cpu& cpu) override {
    return std::make_unique<OoneAgent>(&this->enclave_, cpu, scheduler_.get());
  }

  void RpcHandler(int64_t req, const AgentRpcArgs& args,
                  AgentRpcResponse& response) override {
    switch (req) {
      case OoneScheduler::kDebugRunqueue:
        scheduler_->debug_runqueue_ = true;
        response.response_code = 0;
        return;
      case OoneScheduler::kCountAllTasks:
        response.response_code = scheduler_->CountAllTasks();
        return;
      default:
        response.response_code = -1;
        return;
    }
  }

 private:
  std::unique_ptr<OoneScheduler> scheduler_;
};

}  // namespace ghost

#endif  // GHOST_SCHEDULERS_Oone_Oone_SCHEDULER_H
