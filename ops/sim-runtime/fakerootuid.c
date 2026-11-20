/*
 * fakerootuid.c - make a root process look like an unprivileged one to the programs it spawns.
 *
 * Why this exists
 * ---------------
 * The UE4 simulation binaries shipped in `envs/closeloop_envs/<Scene>/Binaries/Linux/<Scene>`
 * check the effective user id at startup and print
 *
 *     Refusing to run with the root privileges.
 *
 * then exit, when `geteuid()` returns 0.  This machine only has a root account, so the
 * simulator server (`airsim_plugin/AirVLNSimulatorServerTool.py`) can never start its UE4
 * child without help.
 *
 * This shim only *reports* a non-root id; it does not change any real permission, capability
 * or file ownership.  It is inherited by the whole process tree of whatever it is preloaded
 * into, which is exactly what the simulator server needs:
 *
 *     LD_PRELOAD=/path/to/libfakerootuid.so python AirVLNSimulatorServerTool.py --port 30000 ...
 *
 * Do NOT preload it into Xvfb: Xvfb uses the real uid for its socket/auth setup and fails to
 * start when it believes it is uid 1000 (this was measured on this machine).
 *
 * Build (no dependencies):
 *     gcc -shared -fPIC -O2 -o lib/libfakerootuid.so fakerootuid.c
 */
#define _GNU_SOURCE
#include <sys/types.h>
#include <unistd.h>

#ifndef FAKE_ID
#define FAKE_ID 1000
#endif

uid_t getuid(void) { return (uid_t)FAKE_ID; }

uid_t geteuid(void) { return (uid_t)FAKE_ID; }

gid_t getgid(void) { return (gid_t)FAKE_ID; }

gid_t getegid(void) { return (gid_t)FAKE_ID; }
