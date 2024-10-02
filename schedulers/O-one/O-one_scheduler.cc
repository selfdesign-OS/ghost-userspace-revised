// Copyright 2021 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "schedulers/O-one/O-one_scheduler.h"
#include "absl/time/time.h"  // 필요 시 absl 시간 라이브러리 포함
#include <memory>

namespace ghost {

RoundRobinScheduler::RoundRobinScheduler(Enclave* enclave, CpuList cpulist,
                             std::shared_ptr<TaskAllocator<RoundRobinTask>> allocator)
    : BasicDispatchScheduler(enclave, std::move(cpulist),
                             std::move(allocator)) {
  for (const Cpu& cpu : cpus()) {
    int node = 0;
    CpuState* cs = cpu_state(cpu);
    cs->channel = enclave->MakeChannel(GHOST_MAX_QUEUE_ELEMS, node,
                                       MachineTopology()->ToCpuList({cpu}));
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
      cs->active_queue.Empty()) {  // run_queue -> active_queue로 변경
    return;
  }

  const RoundRobinTask* current = cs->current;
  // 우선순위 큐에서 태스크를 가져와서 상태를 덤프
  for (int i = 0; i < 140; ++i) {
    if (!cs->active_queue.priority_queues[i].empty()) {
      absl::FPrintF(stderr, "SchedState[%d]: %s rq_l=%lu\n", cpu.id(),
                    current ? current->gtid.describe() : "none",
                    cs->active_queue.priority_queues[i].size());
    }
  }
}


void RoundRobinScheduler::EnclaveReady() {
  for (const Cpu& cpu : cpus()) {
    CpuState* cs = cpu_state(cpu);
    Agent* agent = enclave()->GetAgent(cpu);

    while (!cs->channel->AssociateTask(agent->gtid(), agent->barrier(),
                                       /*status=*/nullptr)) {
      CHECK_EQ(errno, ESTALE);
    }
  }
}

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

  cs->active_queue.Enqueue(task);

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
  }
}

void RoundRobinScheduler::TaskRunnable(RoundRobinTask* task, const Message& msg) {
  const ghost_msg_payload_task_wakeup* payload =
      static_cast<const ghost_msg_payload_task_wakeup*>(msg.payload());

  CHECK(task->blocked());
  task->run_state = RoundRobinTaskState::kRunnable;

  task->prio_boost = !payload->deferrable;

  if (task->cpu < 0) {
    Cpu cpu = AssignCpu(task);
    Migrate(task, cpu, msg.seqnum());
  } else {
    CpuState* cs = cpu_state_of(task);
    cs->active_queue.Enqueue(task);
  }
}

void RoundRobinScheduler::TaskDeparted(RoundRobinTask* task, const Message& msg) {
  const ghost_msg_payload_task_departed* payload =
      static_cast<const ghost_msg_payload_task_departed*>(msg.payload());

  if (task->oncpu() || payload->from_switchto) {
    TaskOffCpu(task, /*blocked=*/false, payload->from_switchto);
  } else if (task->queued()) {
    CpuState* cs = cpu_state_of(task);
    cs->active_queue.Erase(task);
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
  cs->active_queue.Enqueue(task);

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
  cs->active_queue.Enqueue(task);

  if (payload->from_switchto) {
    Cpu cpu = topology()->cpu(payload->cpu);
    enclave()->GetAgent(cpu)->Ping();
  }
}

void RoundRobinScheduler::TaskSwitchto(RoundRobinTask* task, const Message& msg) {
  TaskOffCpu(task, /*blocked=*/true, /*from_switchto=*/false);
}

void RoundRobinScheduler::TaskOffCpu(RoundRobinTask* task, bool blocked, bool from_switchto) {
  GHOST_DPRINT(3, stderr, "Task %s offcpu %d", task->gtid.describe(), task->cpu);
  CpuState* cs = cpu_state_of(task);

  if (task->oncpu()) {
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    CHECK(from_switchto);
    CHECK_EQ(task->run_state, RoundRobinTaskState::kBlocked);
  }

  task->run_state = blocked ? RoundRobinTaskState::kBlocked : RoundRobinTaskState::kRunnable;

  // 태스크가 blocked 상태가 아니면 다시 큐에 삽입
  if (!blocked) {

      cs->expired_queue.Enqueue(task);  // 만료 큐로 이동
  }
}


void RoundRobinScheduler::TaskOnCpu(RoundRobinTask* task, Cpu cpu) {
  CpuState* cs = cpu_state(cpu);
  cs->current = task;

  GHOST_DPRINT(1, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = RoundRobinTaskState::kOnCpu;
  task->cpu = cpu.id();
  task->preempted = false;
  task->prio_boost = false;

  // 더 이상 active_queue에 있지 않으므로 해당 task를 active_queue에서 제거
  cs->active_queue.Erase(task);
}




// void RoundRobinScheduler::RoundRobinSchedule(const Cpu& cpu, BarrierToken agent_barrier, bool prio_boost) {
//   CpuState* cs = cpu_state(cpu);
//   RoundRobinTask* next = cs->run_queue.Dequeue();  // Fetch task in round-robin order

//   if (!next) {
//     // No task to run, set CPU to idle
//     enclave()->GetRunRequest(cpu)->LocalYield(agent_barrier, 0);
//     return;
//   }

//   // Ensure the task is not already running on another CPU
//   while (next->status_word.on_cpu()) {
//     Pause();  // Wait until task is off the CPU
//   }

//   RunRequest* req = enclave()->GetRunRequest(cpu);
//   req->Open({
//       .target = next->gtid,
//       .target_barrier = next->seqnum,
//       .agent_barrier = agent_barrier,
//       .commit_flags = COMMIT_AT_TXN_COMMIT,
//   });

//   if (req->Commit()) {
//     // Set a time slice of 100 milliseconds for the round-robin task
//     const absl::Duration time_slice = absl::Milliseconds(10);

//     // Task is successfully scheduled on the CPU
//     TaskOnCpu(next, cpu);

//     // Sleep for the duration of the time slice before swapping tasks
//     absl::SleepFor(time_slice);

//     // Trigger CPU task replacement via Ping
//     enclave()->GetAgent(cpu)->Ping();
//   } else {
//     // Scheduling failed, mark task as off the CPU if it was current
//     if (next == cs->current) {
//       TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
//     }

//     // Requeue task and set priority boost
//     next->prio_boost = true;
//     cs->run_queue.Enqueue(next);
//   }
// }


void RoundRobinScheduler::RoundRobinSchedule(const Cpu& cpu, BarrierToken agent_barrier, bool prio_boost) {
  CpuState* cs = cpu_state(cpu);

  // 활성 큐와 만료 큐 교체 필요 시 교체
  cs->active_queue.SwitchQueuesIfNeeded(cs);

  // 우선순위가 높은 큐부터 태스크를 가져옴
  RoundRobinTask* next = nullptr;
  for (int i = 0; i < 140; ++i) {
    if (!cs->active_queue.priority_queues[i].empty()) {
      next = cs->active_queue.priority_queues[i].front();
      cs->active_queue.priority_queues[i].pop_front();
      break;
    }
  }

  if (!next) {
    // 실행할 태스크가 없으면 CPU를 idle 상태로 설정
    enclave()->GetRunRequest(cpu)->LocalYield(agent_barrier, 0);
    return;
  }

  // 태스크가 다른 CPU에서 실행 중인 경우를 대비한 선점 처리
  while (next->status_word.on_cpu()) {
    Pause();
  }

  RunRequest* req = enclave()->GetRunRequest(cpu);
  req->Open({
      .target = next->gtid,
      .target_barrier = next->seqnum,
      .agent_barrier = agent_barrier,
      .commit_flags = COMMIT_AT_TXN_COMMIT,
  });

  if (req->Commit()) {
    TaskOnCpu(next, cpu);
    absl::SleepFor(absl::Milliseconds(10));  // 시간 할당량 조절
    enclave()->GetAgent(cpu)->Ping();  // 다음 태스크로 교체
  } else {
    if (next == cs->current) {
      TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/false);
    }
    cs->active_queue.priority_queues[next->priority].push_back(next);
  }
}









void RoundRobinScheduler::Schedule(const Cpu& cpu, const StatusWord& agent_sw) {
  BarrierToken agent_barrier = agent_sw.barrier();
  CpuState* cs = cpu_state(cpu);

  GHOST_DPRINT(3, stderr, "Schedule: agent_barrier[%d] = %d\n", cpu.id(),
               agent_barrier);

  Message msg;
  while (!(msg = Peek(cs->channel.get())).empty()) {
    DispatchMessage(msg);
    Consume(cs->channel.get(), msg);
  }

  RoundRobinSchedule(cpu, agent_barrier, agent_sw.boosted_priority());
}

void O1PriorityQueue::Enqueue(RoundRobinTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, RoundRobinTaskState::kRunnable);

  task->run_state = RoundRobinTaskState::kQueued;


  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    priority_queues[0].push_front(task);
  else
    priority_queues[0].push_back(task);
}

RoundRobinTask* O1PriorityQueue::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (priority_queues[0].empty()) return nullptr;

  RoundRobinTask* task = priority_queues[0].front();
  CHECK(task->queued());
  task->run_state = RoundRobinTaskState::kRunnable;
  priority_queues[0].pop_front();
  return task;
}

// void O1PriorityQueue::EnqueueTask(RoundRobinTask* task) {
//   // 태스크를 우선순위에 맞게 활성 큐에 삽입
//   int priority = task->priority; 
//   cpu_state_of(task)->active_queue.priority_queues[priority].push_back(task);
// }

// 우선순위가 높은 큐에서 Task를 선택하는 함수
// RoundRobinTask* O1PriorityQueue::DequeueTask() {
//   for (int priority = 0; priority < 140; ++priority) {
//     if (!cpu_state()->active_queue.priority_queues[priority].empty()) {
//       RoundRobinTask* task = cpu_state()->active_queue.priority_queues[priority].front();
//       cpu_state()->active_queue.priority_queues[priority].pop_front();
//       return task;
//     }
//   }
//   return nullptr;  // 실행할 Task가 없는 경우
// }
void O1PriorityQueue::SwitchQueuesIfNeeded(RoundRobinScheduler::CpuState* cs) {
    bool active_empty = true;

    // 활성 큐가 비었는지 확인
    for (int i = 0; i < 140; ++i) {
        if (!cs->active_queue.priority_queues[i].empty()) {
            active_empty = false;
            break;
        }
    }

    // 활성 큐가 비어 있으면 만료 큐와 교체
    if (active_empty) {
        swap(cs->active_queue, cs->expired_queue);  // 두 큐를 교체
    }
}







void O1PriorityQueue::Erase(RoundRobinTask* task) {
  CHECK_EQ(task->run_state, RoundRobinTaskState::kQueued);
  absl::MutexLock lock(&mu_);
  size_t size = priority_queues[0].size();
  if (size > 0) {
    size_t pos = size - 1;
    if (priority_queues[0][pos] == task) {
      priority_queues[0].erase(priority_queues[0].cbegin() + pos);
      task->run_state = RoundRobinTaskState::kRunnable;
      return;
    }

    for (pos = 0; pos < size - 1; pos++) {
      if (priority_queues[0][pos] == task) {
        priority_queues[0].erase(priority_queues[0].cbegin() + pos);
        task->run_state =  RoundRobinTaskState::kRunnable;
        return;
      }
    }
  }
  CHECK(false);
}

std::unique_ptr<RoundRobinScheduler> MultiThreadedRRScheduler(Enclave* enclave,
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
      default:
      return os << "UnknownState";  // Handle any unexpected cases
  }
}

}  //  namespace ghost