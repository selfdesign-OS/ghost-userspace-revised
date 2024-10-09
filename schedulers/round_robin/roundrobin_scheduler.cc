// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "schedulers/round_robin/roundrobin_scheduler.h"
#include "absl/time/time.h"  // 필요 시 absl 시간 라이브러리 포함

#include <memory>

namespace ghost {

RoundRobinScheduler::RoundRobinScheduler(Enclave* enclave, CpuList cpulist,
                             std::shared_ptr<TaskAllocator<RoundRobinTask>> allocator)
    : BasicDispatchScheduler(enclave, std::move(cpulist),
                             std::move(allocator)) {
  for (const Cpu& cpu : cpus()) {
    // TODO: extend Cpu to get numa node.
    int node = 0;
    CpuState* cs = cpu_state(cpu);
    cs->channel = enclave->MakeChannel(GHOST_MAX_QUEUE_ELEMS, node,
                                       MachineTopology()->ToCpuList({cpu}));
    // This channel pointer is valid for the lifetime of RoundRobinScheduler
    if (!default_channel_) {
      default_channel_ = cs->channel.get();
    }
  }
}

void RoundRobinScheduler::DumpAllTasks() {
  fprintf(stderr, "task        state   cpu\n");
  allocator()->ForEachTask([](Gtid gtid, const RoundRobinTask* task) {
    absl::FPrintF(stderr, "%-12s%-8d%-8d%c%c\n", gtid.describe(),
                  task->run_state, task->cpu, task->preempted ? 'P' : '-',
                  task->prio_boost ? 'B' : '-');
    return true;
  });
}

void RoundRobinScheduler::DumpState(const Cpu& cpu, int flags) {
  if (flags & Scheduler::kDumpAllTasks) {
    DumpAllTasks();
  }

  CpuState* cs = cpu_state(cpu);
  if (!(flags & Scheduler::kDumpStateEmptyRQ) && !cs->current &&
      cs->run_queue.Empty()) {
    return;
  }

  const RoundRobinTask* current = cs->current;
  const RoundRobinRq* rq = &cs->run_queue;
  absl::FPrintF(stderr, "SchedState[%d]: %s rq_l=%lu\n", cpu.id(),
                current ? current->gtid.describe() : "none", rq->Size());
}

void RoundRobinScheduler::EnclaveReady() {
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
Cpu RoundRobinScheduler::AssignCpu(RoundRobinTask* task) {
  static auto begin = cpus().begin();
  static auto end = cpus().end();
  static auto next = end;

  if (next == end) {
    next = begin;
  }
  return next++;
}

void RoundRobinScheduler::Migrate(RoundRobinTask* task, Cpu cpu, BarrierToken seqnum) {
  CHECK_EQ(task->run_state, RoundRobinTaskState::kRunnable);
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

void RoundRobinScheduler::TaskNew(RoundRobinTask* task, const Message& msg) {
  const ghost_msg_payload_task_new* payload =
      static_cast<const ghost_msg_payload_task_new*>(msg.payload());

  task->seqnum = msg.seqnum();
  task->run_state = RoundRobinTaskState::kBlocked;

  if (payload->runnable) {
    task->run_state = RoundRobinTaskState::kRunnable;
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    // Wait until task becomes runnable to avoid race between migration
    // and MSG_TASK_WAKEUP showing up on the default channel.
  }
}

void RoundRobinScheduler::TaskRunnable(RoundRobinTask* task, const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());

  CHECK(task->blocked());
  task->run_state = RoundRobinTaskState::kRunnable;

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

void RoundRobinScheduler::TaskDeparted(RoundRobinTask* task, const Message& msg) {
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

void RoundRobinScheduler::TaskDead(RoundRobinTask* task, const Message& msg) {
  CHECK(task->blocked());
  allocator()->FreeTask(task);
}

void RoundRobinScheduler::TaskYield(RoundRobinTask* task, const Message& msg) {
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

void RoundRobinScheduler::TaskBlocked(RoundRobinTask* task, const Message& msg) {
  const ghost_msg_payload_task_blocked* payload =
      static_cast<const ghost_msg_payload_task_blocked*>(msg.payload());

  TaskOffCpu(task, /*blocked=*/true, payload->from_switchto);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void RoundRobinScheduler::TaskPreempted(RoundRobinTask* task, const Message& msg) {
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

void RoundRobinScheduler::TaskSwitchto(RoundRobinTask* task, const Message& msg) {
  TaskOffCpu(task, /*blocked=*/true, /*from_switchto=*/false);
}

void RoundRobinScheduler::CheckTick(const Cpu& cpu) // 왜 만들었지
  ABSL_NO_THREAD_SAFETY_ANALYSIS {
  CpuState* cs = cpu_state(cpu);
  //cs->run_queue.mu_.AssertHeld();

  if (cs->current) {
    // If we were on cpu, check if we have run for longer than
    // Granularity(). If so, force picking another task via setting current
    // to nullptr.
    if (absl::Nanoseconds(cs->current->status_word.runtime() -
                          cs->current->runtime_at_first_pick_ns) >
     absl::Duration(absl::Nanoseconds(10000000))) {
      //cs->preempt_curr = true;
    }
  }
}


void RoundRobinScheduler::TaskOffCpu(RoundRobinTask* task, bool blocked,
                               bool from_switchto) {
  GHOST_DPRINT(3, stderr, "Task %s offcpu %d", task->gtid.describe(),
               task->cpu);
  CpuState* cs = cpu_state_of(task);

  if (task->oncpu()) {
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    CHECK(from_switchto);
    CHECK_EQ(task->run_state, RoundRobinTaskState::kBlocked);
  }

  task->run_state =
      blocked ? RoundRobinTaskState::kBlocked : RoundRobinTaskState::kRunnable;
}

void RoundRobinScheduler::TaskOnCpu(RoundRobinTask* task, Cpu cpu) {
  CpuState* cs = cpu_state(cpu);
  cs->current = task;

  GHOST_DPRINT(3, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = RoundRobinTaskState::kOnCpu;
  task->cpu = cpu.id();
  task->preempted = false;
  task->prio_boost = false;
}

void RoundRobinScheduler::RoundRobinSchedule(const Cpu& cpu, BarrierToken agent_barrier,
                                 bool prio_boost) {
  printf("schedule start\n");
  CpuState* cs = cpu_state(cpu);
  RoundRobinTask* next = nullptr;

  

  if (!prio_boost) { // 우선순위 없다면
    printf("not prio_boost\n");
    next = cs->current; // 현재꺼 그대로 실행
    
    if (!next){ // 작업 끝났다면 다음 작업 가져오기
      printf("fetch next task\n");
      //next = cs->run_queue.PickNextTask(prev, allocator(), cs);
      next = cs->run_queue.Dequeue();

      if(!next){ // null이면 return
        return;
      }
      // Todo : 누적 시간 어케 볼건지
      if(!next->runtime_change){
        next->runtime_at_first_pick_ns = next->status_word.runtime(); // 처음 시간 체크
        next->runtime_change = true;
      }
    }

    printf("next->status_word.runtime: %lu\n", next->status_word.runtime());
    printf("next->runtime_at_first_pick_ns: %lu\n", next->runtime_at_first_pick_ns);
    printf("runtime_diff: %lu\n", absl::Nanoseconds(next->status_word.runtime() - next->runtime_at_first_pick_ns));

    if(absl::Nanoseconds(next->status_word.runtime() - next->runtime_at_first_pick_ns)
     >= absl::Nanoseconds(10000000)){ // 타임 슬라이스 초과
      // 현재 작업 뒤로 보내고 다음 작업 실행시키기
      printf("time slice bump\n");
      if (next == cs->current) {
        TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
      }
      next->runtime_at_first_pick_ns = 0;
      next->runtime_change = false;

      // boost X => 큐 뒤에 넣음
      printf("enqueue\n");
      cs->run_queue.Enqueue(next);
      return;
    }    
     
  }

  GHOST_DPRINT(3, stderr, "RoundRobinSchedule %s on %s cpu %d ",
               next ? next->gtid.describe() : "idling",
               prio_boost ? "prio-boosted" : "", cpu.id());


  RunRequest* req = enclave()->GetRunRequest(cpu);
  if (next) {
    printf("next gogo\n");
    // 'next'가 CPU에서 벗어나기를 기다린 후에 다른 작업으로 전환.
    // 이것은 불필요해 보일 수 있지만, 작업이 CPU에 최초로 할당된 이후로
    // 다른 CPU로 작업을 이동시키지 않기 때문입니다. 하지만 SwitchTo
    // 대상으로 지정된 작업은 에이전트가 인식하지 못한 채 다른 CPU에서
    // 실행될 수 있습니다. 이 경우 SwitchTo 대상이 차단된 상태라면,
    // 에이전트가 이를 감지하지 못하고 실행 큐에 넣지 않습니다.
    //
    // 하지만 'next'가 SwitchTo 체인의 마지막 작업이라면, 'next'가 원격
    // CPU에서 완전히 벗어나기 전에 TASK_WAKEUP 메시지를 처리할 수 있습니다.
    // 아래의 'on_cpu()' 확인은 이 시나리오를 처리합니다.
    //
    // 더 자세한 내용은 'go/switchto-ghost'를 참조하세요.
    while (next->status_word.on_cpu()) {
      Pause(); // 'next' 작업이 CPU에서 벗어날 때까지 대기
    }

    req->Open({
        .target = next->gtid,
        .target_barrier = next->seqnum,
        .agent_barrier = agent_barrier,
        .commit_flags = COMMIT_AT_TXN_COMMIT,
    });

    if (req->Commit()) {
      // 트랜잭션 커밋 성공, 'next' 작업이 CPU에서 실행됨.
      printf("success commit\n");
      TaskOnCpu(next, cpu);

    } else {
      GHOST_DPRINT(3, stderr, "RoundRobinSchedule: commit failed (state=%d)",
                   req->state());

      if (next == cs->current) {
        TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
      }

      // 트랜잭션 커밋이 실패했으므로 'next' 작업을 실행 큐의 앞에 다시 넣음.
      next->prio_boost = true;
      cs->run_queue.Enqueue(next);
    }
  } else {
    printf("no next\n");
    // 만약 prio_boost로 인해 LocalYield가 호출된 경우, CPU가 유휴 상태일 때
    // 커널이 에이전트에게 제어를 다시 반환하도록 지시.
    int flags = 0;
    if (prio_boost && (cs->current || !cs->run_queue.Empty())) {
      flags = RTLA_ON_IDLE;
    }
    req->LocalYield(agent_barrier, flags);
  }


}

void RoundRobinScheduler::Schedule(const Cpu& cpu, const StatusWord& agent_sw) {
  BarrierToken agent_barrier = agent_sw.barrier();
  CpuState* cs = cpu_state(cpu);

  GHOST_DPRINT(3, stderr, "Schedule: agent_barrier[%d] = %d\n", cpu.id(),
               agent_barrier);

  //printf("msg pick start\n"); // 왜 printf 넣으면 돌아가고 안 넣으면 안 돌아감?
  Message msg;
  while (!(msg = Peek(cs->channel.get())).empty()) {
    DispatchMessage(msg);
    Consume(cs->channel.get(), msg);
  }
  //printf("msg pick end\n");

  RoundRobinSchedule(cpu, agent_barrier, agent_sw.boosted_priority());
}

void RoundRobinRq::Enqueue(RoundRobinTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, RoundRobinTaskState::kRunnable);

  task->run_state = RoundRobinTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    rq_.push_front(task);
  else
    rq_.push_back(task);
}

RoundRobinTask* RoundRobinRq::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (rq_.empty()) return nullptr;

  RoundRobinTask* task = rq_.front();
  CHECK(task->queued());
  task->run_state = RoundRobinTaskState::kRunnable;
  rq_.pop_front();
  return task;
}

void RoundRobinRq::Erase(RoundRobinTask* task) {
  CHECK_EQ(task->run_state, RoundRobinTaskState::kQueued);
  absl::MutexLock lock(&mu_);
  size_t size = rq_.size();
  if (size > 0) {
    // Check if 'task' is at the back of the runqueue (common case).
    size_t pos = size - 1;
    if (rq_[pos] == task) {
      rq_.erase(rq_.cbegin() + pos);
      task->run_state = RoundRobinTaskState::kRunnable;
      return;
    }

    // Now search for 'task' from the beginning of the runqueue.
    for (pos = 0; pos < size - 1; pos++) {
      if (rq_[pos] == task) {
        rq_.erase(rq_.cbegin() + pos);
        task->run_state =  RoundRobinTaskState::kRunnable;
        return;
      }
    }
  }
  CHECK(false);
}

std::unique_ptr<RoundRobinScheduler> MultiThreadedRoundRobinScheduler(Enclave* enclave,
                                                          CpuList cpulist) {
  auto allocator = std::make_shared<ThreadSafeMallocTaskAllocator<RoundRobinTask>>();
  auto scheduler = std::make_unique<RoundRobinScheduler>(enclave, std::move(cpulist),
                                                   std::move(allocator));
  return scheduler;
}

void RoundRobinAgent::AgentThread() {
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

std::ostream& operator<<(std::ostream& os, const RoundRobinTaskState& state) {
  switch (state) {
    case RoundRobinTaskState::kBlocked:
      return os << "kBlocked";
    case RoundRobinTaskState::kRunnable:
      return os << "kRunnable";
    case RoundRobinTaskState::kQueued:
      return os << "kQueued";
    case RoundRobinTaskState::kOnCpu:
      return os << "kOnCpu";
  }
}

}  //  namespace ghost
