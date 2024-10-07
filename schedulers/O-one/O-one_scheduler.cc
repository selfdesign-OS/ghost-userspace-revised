// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "schedulers/O-one/O-one_scheduler.h"

#include <memory>

namespace ghost {

FifoScheduler::FifoScheduler(Enclave* enclave, CpuList cpulist,
                             std::shared_ptr<TaskAllocator<FifoTask>> allocator)
    : BasicDispatchScheduler(enclave, std::move(cpulist),
                             std::move(allocator)) {
  for (const Cpu& cpu : cpus()) {
    // TODO: extend Cpu to get numa node.
    int node = 0;
    CpuState* cs = cpu_state(cpu);
    cs->channel = enclave->MakeChannel(GHOST_MAX_QUEUE_ELEMS, node,
                                       MachineTopology()->ToCpuList({cpu}));
    // This channel pointer is valid for the lifetime of FifoScheduler
    if (!default_channel_) {
      default_channel_ = cs->channel.get();
    }
      // CPU Tick 메시지 수신을 위해 각 CPU의 채널을 설정합니다.
  }

}

void FifoScheduler::DumpAllTasks() {
  fprintf(stderr, "task        state   cpu\n");
  allocator()->ForEachTask([](Gtid gtid, const FifoTask* task) {
    absl::FPrintF(stderr, "%-12s%-8d%-8d%c%c\n", gtid.describe(),
                  task->run_state, task->cpu, task->preempted ? 'P' : '-',
                  task->prio_boost ? 'B' : '-');
    return true;
  });
}

void FifoScheduler::DumpState(const Cpu& cpu, int flags) {
  if (flags & Scheduler::kDumpAllTasks) {
    DumpAllTasks();
  }

  CpuState* cs = cpu_state(cpu);
  if (!(flags & Scheduler::kDumpStateEmptyRQ) && !cs->current &&
      cs->run_queue.Empty()) {
    return;
  }

  const FifoTask* current = cs->current;
  const FifoRq* rq = &cs->run_queue;
  absl::FPrintF(stderr, "SchedState[%d]: %s rq_l=%lu\n", cpu.id(),
                current ? current->gtid.describe() : "none", rq->Size());
}

void FifoScheduler::EnclaveReady() {
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    Agent* agent = enclave()->GetAgent(cpu);

    // AssociateTask may fail if agent barrier is stale.
    while (!cs->channel->AssociateTask(agent->gtid(), agent->barrier(),
                                       /*status=*/nullptr)) {
      CHECK_EQ(errno, ESTALE);
    }
  }
  enclave()->SetDeliverTicks(true);

}
void FifoScheduler::CpuTick(const Message& msg) {
  Cpu cpu = topology()->cpu(msg.cpu());
  CpuState* cs = cpu_state(cpu);
    absl::MutexLock lock(&cs->run_queue.mu_);  // 뮤텍스 잠금
  CheckPreemptTick(cpu);

}
void FifoScheduler::CheckPreemptTick(const Cpu& cpu) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  CpuState* cs = cpu_state(cpu);

  if (cs->current) {
    FifoTask* current_task = cs->current;
    uint64_t current_runtime_ns = current_task->status_word.runtime();
    uint64_t elapsed_time_ns = current_runtime_ns - current_task->runtime_at_last_check_ns;

    // 남은 실행 시간에서 경과된 시간을 뺍니다.
    if (elapsed_time_ns >= current_task->remaining_runtime_ns) {
      current_task->remaining_runtime_ns = 0;
      cs->preempt_curr = true;
    } else {
      current_task->remaining_runtime_ns -= elapsed_time_ns;
    }

    // runtime_at_last_check_ns를 현재 runtime으로 업데이트
    current_task->runtime_at_last_check_ns = current_runtime_ns;
  }
}



// Implicitly thread-safe because it is only called from one agent associated
// with the default queue.
Cpu FifoScheduler::AssignCpu(FifoTask* task) {
  static auto begin = cpus().begin();
  static auto end = cpus().end();
  static auto next = end;

  if (next == end) {
    next = begin;
  }
  return next++;
}

void FifoScheduler::Migrate(FifoTask* task, Cpu cpu, BarrierToken seqnum) {
  CHECK_EQ(task->run_state, FifoTaskState::kRunnable);
  CHECK_EQ(task->cpu, -1);

  CpuState* cs = cpu_state(cpu);
  const Channel* channel = cs->channel.get();
  CHECK(channel->AssociateTask(task->gtid, seqnum, /*status=*/nullptr));

  GHOST_DPRINT(3, stderr, "Migrating task %s to cpu %d", task->gtid.describe(),
               cpu.id());
  task->cpu = cpu.id();

  // Make task visible in the new runqueue *after* changing the association
  // (otherwise the task can get oncpu while producing into the old queue).
  cs->run_queue.Enqueue(task);

  // Get the agent's attention so it notices the new task.
  enclave()->GetAgent(cpu)->Ping();
}

void FifoScheduler::TaskNew(FifoTask* task, const Message& msg) {
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());

  task->seqnum = msg.seqnum();
  task->run_state = FifoTaskState::kBlocked;

  if (payload->runnable) {
    task->run_state = FifoTaskState::kRunnable;
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    // Wait until task becomes runnable to avoid race between migration
    // and MSG_TASK_WAKEUP showing up on the default channel.
  }
}

void FifoScheduler::TaskRunnable(FifoTask* task, const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());

  CHECK(task->blocked());
  task->run_state = FifoTaskState::kRunnable;

  // A non-deferrable wakeup gets the same preference as a preempted task.
  // This is because it may be holding locks or resources needed by other
  // tasks to make progress.
  task->prio_boost = !payload->deferrable;

  if (task->cpu < 0) {
    // There cannot be any more messages pending for this task after a
    // MSG_TASK_WAKEUP (until the agent puts it oncpu) so it's safe to
    // migrate.
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    CpuState* cs = cpu_state_of(task);
    cs->run_queue.Enqueue(task);
  }
}

void FifoScheduler::TaskDeparted(FifoTask* task, const Message& msg) {
  const ghost_msg_payload_task_departed* payload =
      static_cast<const ghost_msg_payload_task_departed*>(msg.payload());

  if (task->oncpu() || payload->from_switchto) {
    TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);
  } else if (task->queued()) {
    CpuState* cs = cpu_state_of(task);
    cs->run_queue.Erase(task);
  } else {
    CHECK(task->blocked());
  }

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }

  allocator()->FreeTask(task);
}

void FifoScheduler::TaskDead(FifoTask* task, const Message& msg) {
  CHECK(task->blocked());
  allocator()->FreeTask(task);
}

void FifoScheduler::TaskYield(FifoTask* task, const Message& msg) {
  const ghost_msg_payload_task_yield* payload =
      static_cast<const ghost_msg_payload_task_yield*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);
  task->remaining_runtime_ns = 0;

  CpuState* cs = cpu_state_of(task);
  cs->run_queue.Enqueue(task);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void FifoScheduler::TaskBlocked(FifoTask* task, const Message& msg) {
  const ghost_msg_payload_task_blocked* payload =
      static_cast<const ghost_msg_payload_task_blocked*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/true, payload->from_switchto);
  task->remaining_runtime_ns = 0;

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void FifoScheduler::TaskPreempted(FifoTask* task, const Message& msg) {
  const ghost_msg_payload_task_preempt* payload =
      static_cast<const ghost_msg_payload_task_preempt*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);

  task->preempted = true;
  task->prio_boost = true;
  CpuState* cs = cpu_state_of(task);
  cs->run_queue.Enqueue(task);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void FifoScheduler::TaskSwitchto(FifoTask* task, const Message& msg) {
  TaskOffCpu(task, /*blocked=*/true, /*from_switchto=*/false);
}


void FifoScheduler::TaskOffCpu(FifoTask* task, bool blocked,
                               bool from_switchto) {
  GHOST_DPRINT(3, stderr, "Task %s offcpu %d", task->gtid.describe(),
               task->cpu);
  CpuState* cs = cpu_state_of(task);

  if (task->oncpu()) {
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else if (task->queued()) {
    // 태스크가 큐에 있는 경우 활성 큐와 만료 큐에서 제거합니다.
    cs->run_queue.Erase(task);
    cs->expired_queue.Erase(task);
  }else {
    CHECK(from_switchto);
    CHECK_EQ(task->run_state, FifoTaskState::kBlocked);
  }

  task->run_state =
      blocked ? FifoTaskState::kBlocked : FifoTaskState::kRunnable;
  // 태스크가 블록된 경우 runtime_at_first_pick_ns를 초기화하도록 설정
  if (blocked) {
    task->reset_runtime = true;
  }
}

void FifoScheduler::TaskOnCpu(FifoTask* task, Cpu cpu) {
  CpuState* cs = cpu_state(cpu);
  cs->current = task;

  GHOST_DPRINT(3, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = FifoTaskState::kOnCpu;
  task->cpu = cpu.id();
  task->preempted = false;
  task->prio_boost = false;

  // 남은 실행 시간이 없으면 새로운 시간 슬라이스를 설정
  if (task->remaining_runtime_ns == 0) {
    task->remaining_runtime_ns = absl::ToInt64Nanoseconds(absl::Milliseconds(5));  // 5ms를 나노초로 변환
  }

  // runtime_at_last_check_ns를 현재 누적 실행 시간으로 설정
  task->runtime_at_last_check_ns = task->status_word.runtime();
}


void FifoRq::swap(FifoRq& other) {
    // To prevent deadlocks, always lock the mutexes in the same order.
    // For example, lock the mutex with the lower memory address first.
    FifoRq* first = this < &other ? this : &other;
    FifoRq* second = this < &other ? &other : this;

    absl::MutexLock lock1(&first->mu_);
    absl::MutexLock lock2(&second->mu_);

    std::swap(this->rq_, other.rq_);
}

void FifoScheduler::FifoSchedule(const Cpu& cpu, BarrierToken agent_barrier, bool prio_boost) {
  CpuState* cs = cpu_state(cpu);
  FifoTask* next = cs->current;

  // 선점이 필요한 경우 처리
  if (cs->preempt_curr) {
    if (next) {
      // 현재 태스크를 적절한 큐로 이동
      TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);

      if (next->remaining_runtime_ns > 0) {
        // 남은 실행 시간이 있으면 활성 큐에 다시 추가
        cs->run_queue.Enqueue(next);
      } else {
        // 실행 시간이 다 소진되었으면 만료 큐에 추가하고 remaining_runtime_ns 초기화
        next->remaining_runtime_ns = 0;
        cs->expired_queue.Enqueue(next);
      }

      cs->current = nullptr;
      next = nullptr;
    }
    cs->preempt_curr = false;  // 선점 플래그 초기화
  }

  if (!next) {
    if (cs->run_queue.Empty() && !cs->expired_queue.Empty()) {
      GHOST_DPRINT(1, stderr, "swap completed");
      cs->run_queue.swap(cs->expired_queue);
    }
    next = cs->run_queue.Dequeue();
  }

  GHOST_DPRINT(3, stderr, "FifoSchedule %s on cpu %d ",
               next ? next->gtid.describe() : "idling",
               cpu.id());

  RunRequest* req = enclave()->GetRunRequest(cpu);
  if (next) {
    while (next->status_word.on_cpu()) {
      Pause();
    }

    req->Open({
        .target = next->gtid,
        .target_barrier = next->seqnum,
        .agent_barrier = agent_barrier,
        .commit_flags = COMMIT_AT_TXN_COMMIT,
    });

    if (req->Commit()) {
      GHOST_DPRINT(3, stderr, "Task %s oncpu %d", next->gtid.describe(), cpu.id());
      TaskOnCpu(next, cpu);
    } else {
      GHOST_DPRINT(3, stderr, "FifoSchedule: commit failed (state=%d)", req->state());
      // Commit 실패 시 태스크를 다시 run_queue에 추가
      cs->run_queue.Enqueue(next);
    }
  } else {
    req->LocalYield(agent_barrier, 0);
  }
}




void FifoScheduler::Schedule(const Cpu& cpu, const StatusWord& agent_sw) {
  BarrierToken agent_barrier = agent_sw.barrier();
  CpuState* cs = cpu_state(cpu);

  GHOST_DPRINT(3, stderr, "Schedule: agent_barrier[%d] = %d\n", cpu.id(),
               agent_barrier);

  Message msg;
  while (!(msg = Peek(cs->channel.get())).empty()) {
    DispatchMessage(msg);
    Consume(cs->channel.get(), msg);
  }

  FifoSchedule(cpu, agent_barrier, agent_sw.boosted_priority());
}

void FifoRq::Enqueue(FifoTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, FifoTaskState::kRunnable);

  task->run_state = FifoTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    rq_.push_front(task);
  else
    rq_.push_back(task);
}

FifoTask* FifoRq::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (rq_.empty()) return nullptr;

  FifoTask* task = rq_.front();
  CHECK(task->queued());
  task->run_state = FifoTaskState::kRunnable;
  rq_.pop_front();
  return task;
}

void FifoRq::Erase(FifoTask* task) {
  CHECK_EQ(task->run_state, FifoTaskState::kQueued);
  absl::MutexLock lock(&mu_);
  size_t size = rq_.size();
  if (size > 0) {
    // Check if 'task' is at the back of the runqueue (common case).
    size_t pos = size - 1;
    if (rq_[pos] == task) {
      rq_.erase(rq_.cbegin() + pos);
      task->run_state = FifoTaskState::kRunnable;
      return;
    }

    // Now search for 'task' from the beginning of the runqueue.
    for (pos = 0; pos < size - 1; pos++) {
      if (rq_[pos] == task) {
        rq_.erase(rq_.cbegin() + pos);
        task->run_state =  FifoTaskState::kRunnable;
        return;
      }
    }
  }
  CHECK(false);
}

std::unique_ptr<FifoScheduler> MultiThreadedFifoScheduler(Enclave* enclave,
                                                          CpuList cpulist) {
  auto allocator = std::make_shared<ThreadSafeMallocTaskAllocator<FifoTask>>();
  auto scheduler = std::make_unique<FifoScheduler>(enclave, std::move(cpulist),
                                                   std::move(allocator));
  return scheduler;
}

void FifoAgent::AgentThread() {
  gtid().assign_name("Agent:" + std::to_string(cpu().id()));
  if (verbose() > 1) {
    printf("Agent tid:=%d\n", gtid().tid());
  }
  SignalReady();
  WaitForEnclaveReady();

  PeriodicEdge debug_out(absl::Seconds(1));

  while (!Finished() || !scheduler_->Empty(cpu())) {
    scheduler_->Schedule(cpu(), status_word());

    if (verbose() && debug_out.Edge()) {
      static const int flags = verbose() > 1 ? Scheduler::kDumpStateEmptyRQ : 0;
      if (scheduler_->debug_runqueue_) {
        scheduler_->debug_runqueue_ = false;
        scheduler_->DumpState(cpu(), Scheduler::kDumpAllTasks);
      } else {
        scheduler_->DumpState(cpu(), flags);
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const FifoTaskState& state) {
  switch (state) {
    case FifoTaskState::kBlocked:
      return os << "kBlocked";
    case FifoTaskState::kRunnable:
      return os << "kRunnable";
    case FifoTaskState::kQueued:
      return os << "kQueued";
    case FifoTaskState::kOnCpu:
      return os << "kOnCpu";
  }
}

}  //  namespace ghost
