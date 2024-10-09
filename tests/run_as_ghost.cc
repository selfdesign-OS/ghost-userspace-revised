#include <stdio.h>
#include <stdlib.h>
#include <atomic>
#include <memory>
#include <vector>
#include "lib/base.h"
#include "lib/ghost.h"

// ghOSt 스케줄러에서 프로그램 실행
namespace ghost {
namespace {

void RunAsGhost(const char* command) {
  GhostThread t(GhostThread::KernelScheduler::kGhost, [command] {
    printf("Running command: %s\n", command);
    int ret = system(command);
    if (ret == -1) {
      perror("Error running the command");
    } else {
      printf("Command completed with exit code: %d\n", WEXITSTATUS(ret));
    }
  });

  t.Join();
}

}  // namespace
}  // namespace ghost

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <command_to_run>\n", argv[0]);
    return 1;
  }

  // 인자로 받은 명령어를 ghOSt 스케줄러에서 실행
  for (int i = 1; i < argc; ++i) {
    printf("Scheduling: %s\n", argv[i]);
    ghost::RunAsGhost(argv[i]);
  }

  return 0;
}
