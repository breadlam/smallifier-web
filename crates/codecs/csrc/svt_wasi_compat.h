/* Force-included when building SVT-AV1 for wasm without threads: the few
 * POSIX calls wasi-libc lacks there. geteuid() reports "not root", so the
 * library never tries to switch to real-time scheduling. */
#pragma once
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>

static inline uid_t geteuid(void) { return 1; }
#ifndef SCHED_FIFO
#define SCHED_FIFO 1
#endif
static inline int pthread_setschedparam(pthread_t t, int policy, const struct sched_param *p) {
    (void)t;
    (void)policy;
    (void)p;
    return 0;
}
