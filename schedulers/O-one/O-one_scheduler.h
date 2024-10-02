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
class RoundRobinScheduler;  // 전방 선언
  struct RoundRobinScheduler::CpuState;  // CpuState의 전방 선언

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
  explicit RoundRobinTask(Gtid RounbRobin_task_gtid, ghost_sw_info sw_info)
      : Task<>(RounbRobin_task_gtid, sw_info) {}
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
  int priority = 0;  // 우선순위 필드 추가, 기본값은 0으로 설정

  // A task's priority is boosted on a kernel preemption or a !deferrable
  // wakeup - basically when it may be holding locks or other resources
  // that prevent other tasks from making progress.
  bool prio_boost = false;
};

// class RoundRobinTaskRq {
//  public:
//   RoundRobinTaskRq() = default;
//   RoundRobinTaskRq(const RoundRobinTaskRq&) = delete;
//   RoundRobinTaskRq& operator=(RoundRobinTaskRq&) = delete;

//   RoundRobinTask* Dequeue();
//   void Enqueue(RoundRobinTask* task);

//   // Erase 'task' from the runqueue.
//   //
//   // Caller must ensure that 'task' is on the runqueue in the first place
//   // (e.g. via task->queued()).
//   void Erase(RoundRobinTask* task);

//   size_t Size() const {
//     absl::MutexLock lock(&mu_);
//     return rq_.size();
//   }

//   bool Empty() const { return Size() == 0; }

//  private:
//   mutable absl::Mutex mu_;
//   std::deque<RoundRobinTask*> rq_ ABSL_GUARDED_BY(mu_);
// };

class O1PriorityQueue {
 public:
  O1PriorityQueue() = default;
  O1PriorityQueue(const O1PriorityQueue&) = delete;
  O1PriorityQueue& operator=(O1PriorityQueue&) = delete;

  RoundRobinTask* Dequeue();  // 우선순위가 가장 높은 큐에서 태스크 제거
  void Enqueue(RoundRobinTask* task);  // 태스크를 적절한 우선순위 큐에 삽입
  void Erase(RoundRobinTask* task);  // 태스크를 특정 큐에서 제거
  void SwitchQueuesIfNeeded(RoundRobinScheduler::CpuState* cs);  // CpuState 인스턴스를 전달받음
  size_t Size() const;  // 전체 큐에 있는 태스크의 총 개수
  bool Empty() const;  // 모든 큐가 비었는지 여부 확인
  std::deque<RoundRobinTask*> priority_queues[140];  // 140개의 우선순위 큐
  // swap 함수 정의
  friend void swap(O1PriorityQueue& first, O1PriorityQueue& second) {
    using std::swap;
    swap(first.priority_queues, second.priority_queues);
  }
 private:
  mutable absl::Mutex mu_;  // 멀티스레드 환경에서의 동기화를 위해 사용
};

class RoundRobinScheduler : public BasicDispatchScheduler<RoundRobinTask> {
 public:
  explicit RoundRobinScheduler(Enclave* enclave, CpuList cpulist,
                         std::shared_ptr<TaskAllocator<RoundRobinTask>> allocator);
  ~RoundRobinScheduler() final {}

  void Schedule(const Cpu& cpu, const StatusWord& sw);

  void EnclaveReady() final;
  Channel& GetDefaultChannel() final { return *default_channel_; };

  bool Empty(const Cpu& cpu) {
    CpuState* cs = cpu_state(cpu);
    return cs->active_queue.Empty();
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

 public:
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

    O1PriorityQueue active_queue;  // 활성 우선순위 큐 (active queue)
  O1PriorityQueue expired_queue;  // 만료 우선순위 큐 (expired queue)
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

std::unique_ptr<RoundRobinScheduler> MultiThreadedRRScheduler(Enclave* enclave,
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
        MultiThreadedRRScheduler(&this->enclave_, *this->enclave_.cpus());
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

#endif  // GHOST_SCHEDULERS_FIFO_FIFO_SCHEDULER_H