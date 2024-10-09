// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "schedulers/oone_ding/per_cpu/oone_scheduler.h"

#include <memory>

namespace ghost {

OoneScheduler::OoneScheduler(Enclave* enclave, CpuList cpulist,
                             std::shared_ptr<TaskAllocator<OoneTask>> allocator)
    : BasicDispatchScheduler(enclave, std::move(cpulist),
                             std::move(allocator)) {
  for (const Cpu& cpu : cpus()) {
    // TODO: extend Cpu to get numa node.
    int node = 0;
    CpuState* cs = cpu_state(cpu);
    cs->channel = enclave->MakeChannel(GHOST_MAX_QUEUE_ELEMS, node,
                                       MachineTopology()->ToCpuList({cpu}));
    // This channel pointer is valid for the lifetime of OoneScheduler
    if (!default_channel_) {
      default_channel_ = cs->channel.get();
    }
  }
}

void OoneScheduler::DumpAllTasks() {
  fprintf(stderr, "task        state   cpu\n");
  allocator()->ForEachTask([](Gtid gtid, const OoneTask* task) {
    absl::FPrintF(stderr, "%-12s%-8d%-8d%c%c\n", gtid.describe(),
                  task->run_state, task->cpu, task->preempted ? 'P' : '-',
                  task->prio_boost ? 'B' : '-');
    return true;
  });
}

void OoneScheduler::DumpState(const Cpu& cpu, int flags) {
  if (flags & Scheduler::kDumpAllTasks) {
    DumpAllTasks();
  }

  CpuState* cs = cpu_state(cpu);
  if (!(flags & Scheduler::kDumpStateEmptyRQ) && !cs->current &&
      cs->run_queue.Empty()) {
    return;
  }

  const OoneTask* current = cs->current;
  const OoneRq* rq = &cs->run_queue;
  absl::FPrintF(stderr, "SchedState[%d]: %s aq_l=%lu\n", cpu.id(),
                current ? current->gtid.describe() : "none", rq->SizeOfAq());
}

void OoneScheduler::EnclaveReady() {
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    Agent* agent = enclave()->GetAgent(cpu);

    // AssociateTask may fail if agent barrier is stale.
    while (!cs->channel->AssociateTask(agent->gtid(), agent->barrier(),
                                       /*status=*/nullptr)) {
      CHECK_EQ(errno, ESTALE);
    }
  }
}

// Implicitly thread-safe because it is only called from one agent associated
// with the default queue.
Cpu OoneScheduler::AssignCpu(OoneTask* task) {
  static auto begin = cpus().begin();
  static auto end = cpus().end();
  static auto next = end;

  if (next == end) {
    next = begin;
  }
  return next++;
}

void OoneScheduler::Migrate(OoneTask* task, Cpu cpu, BarrierToken seqnum) {
  CHECK_EQ(task->run_state, OoneTaskState::kRunnable);
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

void OoneScheduler::TaskNew(OoneTask* task, const Message& msg) {
  // GHOST_DPRINT(1, stderr, "[TaskNew called]: %s", task->gtid.describe());
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());
  task->seqnum = msg.seqnum();
  task->SetTimeSlice();
  task->run_state = OoneTaskState::kBlocked;

  if (payload->runnable) {
    task->run_state = OoneTaskState::kRunnable;
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    // Wait until task becomes runnable to avoid race between migration
    // and MSG_TASK_WAKEUP showing up on the default channel.
  }
  // GHOST_DPRINT(1, stderr, "[TaskNew]: %s", task->gtid.describe());
}

void OoneScheduler::TaskRunnable(OoneTask* task, const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());

  CHECK(task->blocked());
  // GHOST_DPRINT(1, stderr, "[TaskRunnable]: %s", task->gtid.describe());
  task->run_state = OoneTaskState::kRunnable;

  // A non-deferrable wakeup gets the same preference as a preempted task.
  // This is because it may be holding locks or resources needed by other
  // tasks to make progress.
  // task->prio_boost = !payload->deferrable;
  task->prio_boost = false;

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

void OoneScheduler::TaskDeparted(OoneTask* task, const Message& msg) {
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

void OoneScheduler::TaskDead(OoneTask* task, const Message& msg) {
  CHECK(task->blocked());
  allocator()->FreeTask(task);
}

void OoneScheduler::TaskYield(OoneTask* task, const Message& msg) {
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

void OoneScheduler::TaskBlocked(OoneTask* task, const Message& msg) {
  const ghost_msg_payload_task_blocked* payload =
      static_cast<const ghost_msg_payload_task_blocked*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/true, payload->from_switchto);
  // GHOST_DPRINT(1, stderr, "[TaskBlocked]: %s", task->gtid.describe());
  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void OoneScheduler::TaskPreempted(OoneTask* task, const Message& msg) {
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

void OoneScheduler::TaskSwitchto(OoneTask* task, const Message& msg) {
  TaskOffCpu(task, /*blocked=*/true, /*from_switchto=*/false);
}


void OoneScheduler::TaskOffCpu(OoneTask* task, bool blocked,
                               bool from_switchto) {
  GHOST_DPRINT(3, stderr, "Task %s offcpu %d", task->gtid.describe(),
               task->cpu);
  CpuState* cs = cpu_state_of(task);

  if (task->oncpu()) {
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    CHECK(from_switchto);
    CHECK_EQ(task->run_state, OoneTaskState::kBlocked);
  }

  task->run_state =
      blocked ? OoneTaskState::kBlocked : OoneTaskState::kRunnable;
}

void OoneScheduler::TaskOnCpu(OoneTask* task, Cpu cpu) {
  CpuState* cs = cpu_state(cpu);
  cs->current = task;

  GHOST_DPRINT(3, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = OoneTaskState::kOnCpu;
  task->start_time = absl::Now();
  task->cpu = cpu.id();
  task->preempted = false;
  task->prio_boost = false;
}

void OoneScheduler::OoneSchedule(const Cpu& cpu, BarrierToken agent_barrier,
                                 bool prio_boost) {
  CpuState* cs = cpu_state(cpu);
  OoneTask* next = nullptr;

  if (!prio_boost) {
    next = cs->current;

    if (next) {
      absl::Duration exec_time = absl::Now() - next->start_time;
      next->time_slice -= exec_time; // 실행 시간 차감

      GHOST_DPRINT(1, stderr, "[OoneSchedule]: %s, remaining time: %ld", next->gtid.describe(), absl::ToInt64Nanoseconds(next->time_slice));
      
      // expired queue로 이동할지 결정
      if (next->time_slice <= absl::ZeroDuration()) {
        GHOST_DPRINT(1, stderr, "CPU[%d]: %s - time slice expired", cpu.id(), next->gtid.describe());
        next->SetTimeSlice(); // time slice 초기화하는 함수
        cs->run_queue.EnqueueExpired(next);
        next = nullptr;
      } else {
        next->start_time = absl::Now();
      }
    }

    if (!next) next = cs->run_queue.Dequeue();
  }

  GHOST_DPRINT(3, stderr, "OoneSchedule %s on %s cpu %d ",
               next ? next->gtid.describe() : "idling",
               prio_boost ? "prio-boosted" : "", cpu.id());

  RunRequest* req = enclave()->GetRunRequest(cpu);
  if (next) {
    // Wait for 'next' to get offcpu before switching to it. This might seem
    // superfluous because we don't migrate tasks past the initial assignment
    // of the task to a cpu. However a SwitchTo target can migrate and run on
    // another CPU behind the agent's back. This is usually undetectable from
    // the agent's pov since the SwitchTo target is blocked and thus !on_rq.
    //
    // However if 'next' happens to be the last task in a SwitchTo chain then
    // it is possible to process TASK_WAKEUP(next) before it has gotten off
    // the remote cpu. The 'on_cpu()' check below handles this scenario.
    //
    // See go/switchto-ghost for more details.
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
      // Txn commit succeeded and 'next' is oncpu.
      TaskOnCpu(next, cpu);
    } else {
      GHOST_DPRINT(3, stderr, "OoneSchedule: commit failed (state=%d)",
                   req->state());

      if (next == cs->current) {
        TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
      }

      // Txn commit failed so push 'next' to the front of runqueue.
      next->prio_boost = true;
      cs->run_queue.Enqueue(next);
    }
  } else {
    // If LocalYield is due to 'prio_boost' then instruct the kernel to
    // return control back to the agent when CPU is idle.
    int flags = 0;
    if (prio_boost && (cs->current || !cs->run_queue.Empty())) {
      flags = RTLA_ON_IDLE;
    }
    req->LocalYield(agent_barrier, flags);
  }
}

void OoneScheduler::Schedule(const Cpu& cpu, const StatusWord& agent_sw) {
  BarrierToken agent_barrier = agent_sw.barrier();
  CpuState* cs = cpu_state(cpu);

  GHOST_DPRINT(3, stderr, "Schedule: agent_barrier[%d] = %d\n", cpu.id(),
               agent_barrier);

  Message msg;
  while (!(msg = Peek(cs->channel.get())).empty()) {
    DispatchMessage(msg);
    Consume(cs->channel.get(), msg);
  }

  OoneSchedule(cpu, agent_barrier, agent_sw.boosted_priority());
}

void OoneRq::Enqueue(OoneTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, OoneTaskState::kRunnable);

  task->run_state = OoneTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    aq_.push_front(task);
  else
    aq_.push_back(task);
}

void OoneRq::EnqueueExpired(OoneTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, OoneTaskState::kRunnable);

  task->run_state = OoneTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    eq_.push_front(task);
  else
    eq_.push_back(task);
}

OoneTask* OoneRq::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (aq_.empty()) {
    if (eq_.empty()) {
      return nullptr;
    } else {
      GHOST_DPRINT(1, stderr, "[Swap Queue called] aq_size: %d, eq_size: %d", SizeOfAq(), SizeOfEq());
      std::swap(aq_, eq_);
    }
  }

  OoneTask* task = aq_.front();
  CHECK(task->queued());
  task->run_state = OoneTaskState::kRunnable;
  aq_.pop_front();
  return task;
}

void OoneRq::Erase(OoneTask* task) {
  CHECK_EQ(task->run_state, OoneTaskState::kQueued);
  absl::MutexLock lock(&mu_);
  //active queue에서 찾기
  size_t size = aq_.size();
  if (size > 0) {
    // Check if 'task' is at the back of the runqueue (common case).
    size_t pos = size - 1;
    if (aq_[pos] == task) {
      aq_.erase(aq_.cbegin() + pos);
      task->run_state = OoneTaskState::kRunnable;
      return;
    }

    // Now search for 'task' from the beginning of the runqueue.
    for (pos = 0; pos < size - 1; pos++) {
      if (aq_[pos] == task) {
        aq_.erase(aq_.cbegin() + pos);
        task->run_state =  OoneTaskState::kRunnable;
        return;
      }
    }
  }

  // expired queue에서 찾기
  size = eq_.size();
  if (size > 0) {
    size_t pos = size - 1;
    if (eq_[pos] == task) {
      eq_.erase(eq_.cbegin() + pos);
      task->run_state = OoneTaskState::kRunnable;
      return;
    }

    for (pos = 0; pos < size - 1; pos++) {
      if (eq_[pos] == task) {
        eq_.erase(eq_.cbegin() + pos);
        task->run_state = OoneTaskState::kRunnable;
        return;
      }
    }
  }
  // 못 찾은 경우
  CHECK(false);
}

std::unique_ptr<OoneScheduler> MultiThreadedOoneScheduler(Enclave* enclave,
                                                          CpuList cpulist) {
  auto allocator = std::make_shared<ThreadSafeMallocTaskAllocator<OoneTask>>();
  auto scheduler = std::make_unique<OoneScheduler>(enclave, std::move(cpulist),
                                                   std::move(allocator));
  return scheduler;
}

void OoneAgent::AgentThread() {
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

std::ostream& operator<<(std::ostream& os, const OoneTaskState& state) {
  switch (state) {
    case OoneTaskState::kBlocked:
      return os << "kBlocked";
    case OoneTaskState::kRunnable:
      return os << "kRunnable";
    case OoneTaskState::kQueued:
      return os << "kQueued";
    case OoneTaskState::kOnCpu:
      return os << "kOnCpu";
  }
}

}  //  namespace ghost
