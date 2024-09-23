#include <stdio.h>
#include <unistd.h>

int main() {
    int i;
    for (i = 0; i < 10; i++) {
        printf("Process %d running\n", getpid());
        sleep(1); // 1초 대기
    }
    return 0;
}
