// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef GHOST_SCHEDULERS_ROUNDROBIN_SCHEDULER_H
#define GHOST_SCHEDULERS_ROUNDROBIN_SCHEDULER_H

#include <deque>
#include <memory>

#include "lib/agent.h"
#include "lib/scheduler.h"

namespace ghost {

enum class RoundRobinTaskState {
  kBlocked,   // not on runqueue.
  kRunnable,  // transitory state:
              // 1. kBlocked->kRunnable->kQueued
              // 2. kQueued->kRunnable->kOnCpu
  kQueued,    // on runqueue.
  kOnCpu,     // running on cpu.
};

// For CHECK and friends.
std::ostream& operator<<(std::ostream& os, const RoundRobinTaskState& state);

struct RoundRobinTask : public Task<> {
  explicit RoundRobinTask(Gtid roundrobin_task_gtid, ghost_sw_info sw_info)
      : Task<>(roundrobin_task_gtid, sw_info) {}
  ~RoundRobinTask() override {}

  inline bool blocked() const { return run_state == RoundRobinTaskState::kBlocked; }
  inline bool queued() const { return run_state == RoundRobinTaskState::kQueued; }
  inline bool oncpu() const { return run_state == RoundRobinTaskState::kOnCpu; }

  // N.B. _runnable() is a transitory state typically used during runqueue
  // manipulation. It is not expected to be used from task msg callbacks.
  //
  // If you are reading this then you probably want to take a closer look
  // at queued() instead.
  inline bool _runnable() const {
    return run_state == RoundRobinTaskState::kRunnable;
  }

  RoundRobinTaskState run_state = RoundRobinTaskState::kBlocked;
  int cpu = -1;

  // Whether the last execution was preempted or not.
  bool preempted = false;

  // A task's priority is boosted on a kernel preemption or a !deferrable
  // wakeup - basically when it may be holding locks or other resources
  // that prevent other tasks from making progress.
  bool prio_boost = false;

  // 추가
  absl::Time time_slice;
  // runtime_at_first_pick은 해당 태스크가 처음 선택되었을 때의 런타임
  uint64_t runtime_at_first_pick_ns = 0;
  bool runtime_change = false;
};

class RoundRobinRq {
 public:
  RoundRobinRq() = default;
  RoundRobinRq(const RoundRobinRq&) = delete;
  RoundRobinRq& operator=(RoundRobinRq&) = delete;

  RoundRobinTask* Dequeue();
  void Enqueue(RoundRobinTask* task);

  // Erase 'task' from the runqueue.
  //
  // Caller must ensure that 'task' is on the runqueue in the first place
  // (e.g. via task->queued()).
  void Erase(RoundRobinTask* task);
  

  size_t Size() const {
    absl::MutexLock lock(&mu_);
    return rq_.size();
  }

  bool Empty() const { return Size() == 0; }
  mutable absl::Mutex mu_;
  

 private:
  // mutable absl::Mutex mu_;
  std::deque<RoundRobinTask*> rq_ ABSL_GUARDED_BY(mu_);

  //void SwapQueue(RoundRobinRq* rq1, RoundRobinRq* rq2); 

};

class RoundRobinScheduler : public BasicDispatchScheduler<RoundRobinTask> {
 public:
  explicit RoundRobinScheduler(Enclave* enclave, CpuList cpulist,
                         std::shared_ptr<TaskAllocator<RoundRobinTask>> allocator);
  ~RoundRobinScheduler() final {}

  void Schedule(const Cpu& cpu, const StatusWord& sw);
  void CheckTick(const Cpu& cpu);//tickcheck
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
    allocator()->ForEachTask([&num_tasks](Gtid gtid, const RoundRobinTask* task) {
      ++num_tasks;
      return true;
    });
    return num_tasks;
  }

  static constexpr int kDebugRunqueue = 1;
  static constexpr int kCountAllTasks = 2;

 protected:
  void TaskNew(RoundRobinTask* task, const Message& msg) final;
  void TaskRunnable(RoundRobinTask* task, const Message& msg) final;
  void TaskDeparted(RoundRobinTask* task, const Message& msg) final;
  void TaskDead(RoundRobinTask* task, const Message& msg) final;
  void TaskYield(RoundRobinTask* task, const Message& msg) final;
  void TaskBlocked(RoundRobinTask* task, const Message& msg) final;
  void TaskPreempted(RoundRobinTask* task, const Message& msg) final;
  void TaskSwitchto(RoundRobinTask* task, const Message& msg) final;

  
  

 private:
  void RoundRobinSchedule(const Cpu& cpu, BarrierToken agent_barrier,
                    bool prio_boosted);
  void TaskOffCpu(RoundRobinTask* task, bool blocked, bool from_switchto);
  void TaskOnCpu(RoundRobinTask* task, Cpu cpu);
  void Migrate(RoundRobinTask* task, Cpu cpu, BarrierToken seqnum);
  Cpu AssignCpu(RoundRobinTask* task);
  void DumpAllTasks();

  struct CpuState {
    RoundRobinTask* current = nullptr;
    std::unique_ptr<Channel> channel = nullptr;
    RoundRobinRq run_queue; // 활성 큐
    RoundRobinRq expired_queue; // 만료 큐
  } ABSL_CACHELINE_ALIGNED;

  inline CpuState* cpu_state(const Cpu& cpu) { return &cpu_states_[cpu.id()]; }

  inline CpuState* cpu_state_of(const RoundRobinTask* task) {
    CHECK_GE(task->cpu, 0);
    CHECK_LT(task->cpu, MAX_CPUS);
    return &cpu_states_[task->cpu];
  }

  CpuState cpu_states_[MAX_CPUS];
  Channel* default_channel_ = nullptr;
  
};

std::unique_ptr<RoundRobinScheduler> MultiThreadedRoundRobinScheduler(Enclave* enclave,
                                                          CpuList cpulist);
class RoundRobinAgent : public LocalAgent {
 public:
  RoundRobinAgent(Enclave* enclave, Cpu cpu, RoundRobinScheduler* scheduler)
      : LocalAgent(enclave, cpu), scheduler_(scheduler) {}

  void AgentThread() override;
  Scheduler* AgentScheduler() const override { return scheduler_; }

 private:
  RoundRobinScheduler* scheduler_;
};

template <class EnclaveType>
class FullRoundRobinAgent : public FullAgent<EnclaveType> {
 public:
  explicit FullRoundRobinAgent(AgentConfig config) : FullAgent<EnclaveType>(config) {
    scheduler_ =
        MultiThreadedRoundRobinScheduler(&this->enclave_, *this->enclave_.cpus());
    this->StartAgentTasks();
    this->enclave_.Ready();
  }

  ~FullRoundRobinAgent() override {
    this->TerminateAgentTasks();
  }

  std::unique_ptr<Agent> MakeAgent(const Cpu& cpu) override {
    return std::make_unique<RoundRobinAgent>(&this->enclave_, cpu, scheduler_.get());
  }

  void RpcHandler(int64_t req, const AgentRpcArgs& args,
                  AgentRpcResponse& response) override {
    switch (req) {
      case RoundRobinScheduler::kDebugRunqueue:
        scheduler_->debug_runqueue_ = true;
        response.response_code = 0;
        return;
      case RoundRobinScheduler::kCountAllTasks:
        response.response_code = scheduler_->CountAllTasks();
        return;
      default:
        response.response_code = -1;
        return;
    }
  }

 private:
  std::unique_ptr<RoundRobinScheduler> scheduler_;
};

}  // namespace ghost

#endif  // GHOST_SCHEDULERS_ROUNDROBIN_SCHEDULER_H
