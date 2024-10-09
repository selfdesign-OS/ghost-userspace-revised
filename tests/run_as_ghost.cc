#include <stdio.h>
#include <stdlib.h>
#include "lib/base.h"
#include "lib/ghost.h"

// ghOSt 스케줄러에서 프로그램 실행
void RunAsGhost(const char* command) {
  GhostThread t(GhostThread::KernelScheduler::kGhost, [command] {
    // 지정한 프로그램을 ghOSt 스케줄러에서 실행
    int ret = system(command);
    if (ret == -1) {
      perror("Error running the command");
    }
  });

  // 스레드가 종료될 때까지 대기
  t.Join();
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <command_to_run>\n", argv[0]);
    return 1;
  }

  // 인자로 받은 명령어를 ghOSt 스케줄러에서 실행
  for (int i = 1; i < argc; ++i) {
    printf("Running: %s\n", argv[i]);
    RunAsGhost(argv[i]);
  }

  return 0;
}