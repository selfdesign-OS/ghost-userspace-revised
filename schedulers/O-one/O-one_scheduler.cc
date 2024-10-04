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
    cs->tick_config = enclave_->GetDefaultTickConfig();
  }
    enclave_->SetDeliverTicks(true);

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
  absl::MutexLock l(&cs->mutex);  // 필요 시 뮤텍스 잠금

  if (cs->current) {
    FifoTask* task = cs->current;
    uint64_t current_runtime = task->status_word.runtime();
    uint64_t elapsed_time_ns = current_runtime - task->runtime_at_first_pick_ns;

    // 20ms를 초과했는지 확인
    if (absl::Nanoseconds(elapsed_time_ns) >= absl::Milliseconds(20)) {
      // 시간 초과 처리: 태스크를 expired_queue로 이동
      TaskOffCpu(task, /*blocked=*/false, /*from_switchto=*/false);
      CpuState* cs = cpu_state_of(task);
      cs->expired_queue.Enqueue(task);
      cs->current = nullptr;

      // 다음 스케줄링 시 새로운 태스크를 선택하도록 합니다.
    }
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
  } else {
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
  // runtime_at_first_pick_ns를 현재 누적 실행 시간으로 설정
  if (task->reset_runtime) {
    task->runtime_at_first_pick_ns = task->status_word.runtime();
    task->reset_runtime = false;
  }
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
    FifoTask* next = nullptr;

    if (prio_boost) {
        // prio_boost가 true이면 우선순위가 높은 작업을 선택
        if (!cs->run_queue.Empty()) {
            next = cs->run_queue.Dequeue();
        }
    } else {
        // 시간 슬라이스를 확인하고 작업을 선택
        next = cs->current;
        if (next) {
            uint64_t current_runtime = next->status_word.runtime();
            uint64_t elapsed_time_ns = current_runtime - next->runtime_at_first_pick_ns;
            if (absl::Nanoseconds(elapsed_time_ns) >= absl::Milliseconds(10)) {
                // 시간이 초과되면 expired_queue로 이동
                TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
                cs->expired_queue.Enqueue(next);
                next = nullptr;
            }
        }

        if (!next) {
            if (cs->run_queue.Empty() && !cs->expired_queue.Empty()) {
                cs->run_queue.swap(cs->expired_queue);
            }
            next = cs->run_queue.Dequeue();
        }
    }

    // 이후 코드 동일하게 유지
    GHOST_DPRINT(3, stderr, "FifoSchedule %s on %s cpu %d ",
                 next ? next->gtid.describe() : "idling",
                 prio_boost ? "prio-boosted" : "", cpu.id());

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
            next->prio_boost = true;
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
