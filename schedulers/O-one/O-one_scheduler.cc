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
      cs->run_queue.Empty()) {
    return;
  }

  const RoundRobinTask* current = cs->current;
  const RoundRobinTaskRq* rq = &cs->run_queue;
  absl::FPrintF(stderr, "SchedState[%d]: %s rq_l=%lu\n", cpu.id(),
                current ? current->gtid.describe() : "none", rq->Size());
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

  cs->run_queue.Enqueue(task);

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

void RoundRobinScheduler::TaskOffCpu(RoundRobinTask* task, bool blocked,
                               bool from_switchto) {
  GHOST_DPRINT(3, stderr, "Task %s offcpu %d", task->gtid.describe(),
               task->cpu);
  CpuState* cs = cpu_state_of(task);

  if (task->oncpu()) {
    CHECK_EQ(cs->current, task);
    cs->current = nullptr;
  } else {
    // CHECK(from_switchto);
    // CHECK_EQ(task->run_state, RoundRobinTaskState::kBlocked);
  }

  task->run_state =
      blocked ? RoundRobinTaskState::kBlocked : RoundRobinTaskState::kRunnable;
}
void RoundRobinTaskRq::swap(RoundRobinTaskRq& other) {
    // 두 대기열(rq_)을 교환
    std::swap(rq_, other.rq_);
}
void RoundRobinScheduler::TaskOnCpu(RoundRobinTask* task, Cpu cpu) {
  CpuState* cs = cpu_state(cpu);
  cs->current = task;

  GHOST_DPRINT(1, stderr, "Task %s oncpu %d", task->gtid.describe(), cpu.id());

  task->run_state = RoundRobinTaskState::kOnCpu;
  task->cpu = cpu.id();
  task->preempted = false;
  task->prio_boost = false;
}



void RoundRobinScheduler::RoundRobinSchedule(const Cpu& cpu, BarrierToken agent_barrier, bool prio_boost) {
  CpuState* cs = cpu_state(cpu);
  RoundRobinTask* next = nullptr;
  // 우선순위 부스트가 설정되지 않았을 때 현재 작업 또는 다음 작업을 가져옴
    // 우선순위 부스트가 설정되지 않았을 때 현재 작업 또는 다음 작업을 가져옴
    if (!prio_boost) {
        next = cs->current;
        if (!next) {
            if (cs->run_queue.Empty() && !cs->expired_queue.Empty()) {
                cs->run_queue.swap(cs->expired_queue);

                if (cs->run_queue.Empty()) {
                    enclave()->GetRunRequest(cpu)->LocalYield(agent_barrier, 0);
                    return;
                }
            }
            next = cs->run_queue.Dequeue();
        }
    }

  GHOST_DPRINT(3, stderr, "RoundRobinSchedule %s on %s cpu %d ",
               next ? next->gtid.describe() : "idling",
               prio_boost ? "prio-boosted" : "", cpu.id());

  RunRequest* req = enclave()->GetRunRequest(cpu);

  if (next) {
    // 'next' 작업이 이미 다른 CPU에서 실행 중인지 확인
    while (next->status_word.on_cpu()) {
      Pause();  // 다른 CPU에서 내려올 때까지 대기
    }

    req->Open({
        .target = next->gtid,                // 실행할 작업의 글로벌 ID
        .target_barrier = next->seqnum,      // 시퀀스 넘버
        .agent_barrier = agent_barrier,      // 에이전트 배리어
        .commit_flags = COMMIT_AT_TXN_COMMIT // 커밋 플래그 설정
    });

    if (req->Commit()) {
      // 작업이 CPU에 성공적으로 할당됨
      TaskOnCpu(next, cpu);

      // 라운드로빈 작업을 20 밀리초 타임 슬라이스로 실행
      const absl::Duration time_slice = absl::Milliseconds(20);
      absl::SleepFor(time_slice);
      TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/true);
      cs->expired_queue.Enqueue(next);

      // 다음 작업을 위한 Ping
      enclave()->GetAgent(cpu)->Ping();

    } else {
      GHOST_DPRINT(3, stderr, "RoundRobinSchedule: commit failed (state=%d)", req->state());

      // Commit이 실패한 경우, 작업을 다시 큐에 넣고 우선순위 부스트 처리
      if (next == cs->current) {
        TaskOffCpu(next, /*blocked=*/false, /*from_switchto=*/true);
      }

      next->prio_boost = true;
      cs->run_queue.Enqueue(next);
    }
  } else {

    // 작업이 없는 경우 또는 prio_boost일 때 유휴 상태로 전환
    int flags = 0;
    if (prio_boost && (cs->current || !cs->run_queue.Empty())) {
      flags = RTLA_ON_IDLE;  // CPU가 유휴 상태일 때 에이전트로 제어 반환
    }
    req->LocalYield(agent_barrier, flags);
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

void RoundRobinTaskRq::Enqueue(RoundRobinTask* task) {
  CHECK_GE(task->cpu, 0);
  CHECK_EQ(task->run_state, RoundRobinTaskState::kRunnable);

  task->run_state = RoundRobinTaskState::kQueued;

  absl::MutexLock lock(&mu_);
  if (task->prio_boost)
    rq_.push_front(task);
  else
    rq_.push_back(task);
}


RoundRobinTask* RoundRobinTaskRq::Dequeue() {
  absl::MutexLock lock(&mu_);
  if (rq_.empty()) return nullptr;

  RoundRobinTask* task = rq_.front();
  task->run_state = RoundRobinTaskState::kRunnable;
  rq_.pop_front();
  return task;
}

void RoundRobinTaskRq::Erase(RoundRobinTask* task) {
  CHECK_EQ(task->run_state, RoundRobinTaskState::kQueued);
  absl::MutexLock lock(&mu_);
  size_t size = rq_.size();
  if (size > 0) {
    size_t pos = size - 1;
    if (rq_[pos] == task) {
      rq_.erase(rq_.cbegin() + pos);
      task->run_state = RoundRobinTaskState::kRunnable;
      return;
    }

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