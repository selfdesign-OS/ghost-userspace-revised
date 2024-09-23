// cpu_intensive_task.c
#include <stdio.h>

int main() {
    while (1) {
        // 무한 루프로 CPU 점유
        volatile int x = 0;
        for (int i = 0; i < 1000000; ++i) {
            x = x * i;
        }
    }
    return 0;
}
