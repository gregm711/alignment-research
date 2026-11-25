#include <libproc.h>
#include <stdint.h>
#include <unistd.h>

// XNU attributes CPU energy to a process on supported Apple silicon.
int64_t task_energy_nj(void) {
    struct rusage_info_v6 usage = {0};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V6, (rusage_info_t *)&usage) != 0)
        return -1;
    return (int64_t)usage.ri_energy_nj;
}
