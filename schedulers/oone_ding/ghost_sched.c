#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <dlfcn.h>

int sched_setscheduler(pid_t pid, int policy, const struct sched_param *param) {
    // 원래의 sched_setscheduler 함수 호출을 가로채서 SCHED_GHOST로 강제 설정
    static int (*real_sched_setscheduler)(pid_t, int, const struct sched_param *);
    if (!real_sched_setscheduler) {
        real_sched_setscheduler = dlsym(RTLD_NEXT, "sched_setscheduler");
    }
    struct sched_param ghost_param = { .sched_priority = 0 };
    printf("Forcing process to use SCHED_GHOST...\n");
    return real_sched_setscheduler(pid, SCHED_GHOST, &ghost_param);
}