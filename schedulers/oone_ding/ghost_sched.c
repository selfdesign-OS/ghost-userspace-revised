#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <dlfcn.h>

#ifndef SCHED_GHOST
#define SCHED_GHOST 18
#endif

typedef int (*orig_pthread_create_type)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);

int sched_setscheduler(pid_t pid, int policy, const struct sched_param *param) {
    static int (*real_sched_setscheduler)(pid_t, int, const struct sched_param *);
    if (!real_sched_setscheduler) {
        real_sched_setscheduler = dlsym(RTLD_NEXT, "sched_setscheduler");
    }
    struct sched_param ghost_param = { .sched_priority = 0 };
    printf("Forcing process to use SCHED_GHOST...\n");
    return real_sched_setscheduler(pid, SCHED_GHOST, &ghost_param);
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *), void *arg) {
    static orig_pthread_create_type real_pthread_create = NULL;
    if (!real_pthread_create) {
        real_pthread_create = (orig_pthread_create_type)dlsym(RTLD_NEXT, "pthread_create");
    }
    printf("Creating thread with SCHED_GHOST policy...\n");
    int ret = real_pthread_create(thread, attr, start_routine, arg);

    // After thread creation, set it to SCHED_GHOST
    struct sched_param param = { .sched_priority = 0 };
    sched_setscheduler(0, SCHED_GHOST, &param);

    return ret;
}