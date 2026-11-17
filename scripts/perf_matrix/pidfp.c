// pidfp: read the macOS kernel's physical-footprint counters for processes.
//
//   pidfp [-r] pid...
//
// Prints "pid current lifetime_max interval_max" in bytes per line ("pid err" if unreadable).
// These are the same counters `footprint -p` prints as "Footprint" and "phys_footprint_peak",
// and the number iOS Jetsam compares against a WebContent's limit. Reading takes microseconds,
// so it can be polled at 10 Hz without perturbing the browser, unlike `footprint` (0.1-1 s/call).
// lifetime_max is a kernel high-water mark: no spike between samples can be missed.
// -r resets interval_max after reading it, giving exact per-phase peaks.
// -t appends "cpu_ns w:tid=ns,... j:tid=ns,...": total task CPU (dead threads included) and the CPU
// of each live thread named "Wasm Worklist Helper Thread" (w) / "JIT Worklist Helper Thread" (j),
// JSC's background tier-up compilers. Per-thread ids let the reader keep exited threads' time.
#include <libproc.h>
#include <mach/mach_time.h>
#include <sys/proc_info.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

extern int proc_reset_footprint_interval(pid_t pid);

int main(int argc, char **argv) {
  int reset = 0, threads = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-r") == 0) { reset = 1; continue; }
    if (strcmp(argv[i], "-t") == 0) { threads = 1; continue; }
    pid_t pid = (pid_t)atoi(argv[i]);
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&ri) != 0) { printf("%d err\n", pid); continue; }
    printf("%d %llu %llu %llu", pid, ri.ri_phys_footprint, ri.ri_lifetime_max_phys_footprint, ri.ri_interval_max_phys_footprint);
    if (threads) {
      struct proc_taskinfo ti;
      unsigned long long total = 0;
      char wbuf[8192] = "w:", jbuf[8192] = "j:";
      if (proc_pidinfo(pid, PROC_PIDTASKINFO, 0, &ti, sizeof ti) == (int)sizeof ti) {
        // Task totals are Mach absolute-time units (not ns on Apple silicon); thread times are ns.
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        total = (ti.pti_total_user + ti.pti_total_system) * tb.numer / tb.denom;
      }
      static uint64_t ids[8192];
      int n = proc_pidinfo(pid, PROC_PIDLISTTHREADS, 0, ids, sizeof ids) / (int)sizeof(uint64_t);
      for (int k = 0; k < n; k++) {
        struct proc_threadinfo th;
        if (proc_pidinfo(pid, PROC_PIDTHREADINFO, ids[k], &th, sizeof th) <= 0) continue;
        unsigned long long t = th.pth_user_time + th.pth_system_time;
        char *buf = strstr(th.pth_name, "Wasm Worklist") ? wbuf : strstr(th.pth_name, "JIT Worklist") ? jbuf : NULL;
        if (buf && strlen(buf) < sizeof wbuf - 48)
          snprintf(buf + strlen(buf), sizeof wbuf - strlen(buf), "%s%llx=%llu", buf[2] ? "," : "", (unsigned long long)ids[k], t);
      }
      printf(" %llu %s %s", total, wbuf, jbuf);
    }
    printf("\n");
    if (reset) proc_reset_footprint_interval(pid);
  }
  return 0;
}
