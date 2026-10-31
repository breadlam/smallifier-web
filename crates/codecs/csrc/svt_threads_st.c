/* Single-threaded stand-in for SVT-AV1's svt_threads.c, for wasm builds
 * without threads. At lp=1 SVT-AV1 runs all its kernels cooperatively on
 * the calling thread (CONFIG_SINGLE_THREAD_KERNEL), so nothing here may
 * block: a wait that could only be satisfied by another thread is reported
 * as an error instead of hanging. */
#include <stdio.h>
#include <stdlib.h>
#include "svt_threads.h"

typedef struct {
    uint32_t count, max;
} StSemaphore;

void svt_format_thread_name(char *buf, size_t size, const char *prefix, uint32_t index) {
    snprintf(buf, size, "%s%u", prefix, index);
}

EbHandle svt_create_thread(void *thread_function(void *), void *thread_context, const char *name) {
    (void)thread_function;
    (void)thread_context;
    fprintf(stderr, "svt-av1: thread %s requested in a single-threaded build (use lp=1)\n", name ? name : "");
    return NULL;
}

EbErrorType svt_destroy_thread(EbHandle thread_handle) {
    (void)thread_handle;
    return EB_ErrorNone;
}

EbHandle svt_create_semaphore(uint32_t initial_count, uint32_t max_count) {
    StSemaphore *s = malloc(sizeof(*s));
    if (s) {
        s->count = initial_count;
        s->max = max_count;
    }
    return s;
}

EbErrorType svt_post_semaphore(EbHandle semaphore_handle) {
    StSemaphore *s = semaphore_handle;
    if (s->count < s->max) s->count++;
    return EB_ErrorNone;
}

EbErrorType svt_block_on_semaphore(EbHandle semaphore_handle) {
    StSemaphore *s = semaphore_handle;
    if (s->count == 0) {
        fprintf(stderr, "svt-av1: wait on an empty semaphore in a single-threaded build\n");
        return EB_ErrorSemaphoreUnresponsive;
    }
    s->count--;
    return EB_ErrorNone;
}

EbErrorType svt_destroy_semaphore(EbHandle semaphore_handle) {
    free(semaphore_handle);
    return EB_ErrorNone;
}

EbHandle svt_create_mutex(void) {
    return malloc(1);
}

EbErrorType svt_release_mutex(EbHandle mutex_handle) {
    (void)mutex_handle;
    return EB_ErrorNone;
}

EbErrorType svt_block_on_mutex(EbHandle mutex_handle) {
    (void)mutex_handle;
    return EB_ErrorNone;
}

EbErrorType svt_destroy_mutex(EbHandle mutex_handle) {
    free(mutex_handle);
    return EB_ErrorNone;
}

void svt_aom_atomic_set_u32(AtomicVarU32 *var, uint32_t in) {
    var->obj = in;
}

EbErrorType svt_create_cond_var(CondVar *cond_var) {
    cond_var->val = 0;
    return EB_ErrorNone;
}

EbErrorType svt_set_cond_var(CondVar *cond_var, int32_t newval) {
    cond_var->val = newval;
    return EB_ErrorNone;
}

EbErrorType svt_wait_cond_var(CondVar *cond_var, int32_t input) {
    if (cond_var->val == input) {
        fprintf(stderr, "svt-av1: wait on an unchanged condition in a single-threaded build\n");
        return EB_ErrorMutexUnresponsive;
    }
    return EB_ErrorNone;
}

void svt_run_once(OnceType *once_control, OnceFn init_routine) {
    pthread_once(once_control, init_routine);
}
