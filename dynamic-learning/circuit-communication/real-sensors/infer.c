#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { SENSORS = 16, WARM = 8, VOTE = -1, SKILL = -2, SINGLE = -3, REACTIVE = -4 };

static void refresh(const uint8_t history[SENSORS], int weights[SENSORS],
                    int suspects[SENSORS]) {
    for (int j = 0; j < SENSORS; j++) {
        int degree = 0;
        for (int k = 0; k < SENSORS; k++)
            degree += __builtin_popcount((unsigned)(history[j] ^ history[k])) <= 1;
        weights[j] = 720720 / degree; // Exact common denominator for degrees 1 through 16.
        suspects[j] = degree >= 8;
    }
}

int infer(int mode, int n, const double *readings, const double *cut,
          const int8_t *sign, const double *skill, int best, uint8_t *pred) {
    uint8_t history[SENSORS] = {0};
    int weights[SENSORS] = {0}, suspects[SENSORS] = {0};
    int age = 0, flags = 0, count = 0;
    int dynamic = mode >= 0 || mode == REACTIVE;
    if (dynamic) {
        for (int i = 0; i < WARM; i++)
            for (int j = 0; j < SENSORS; j++) {
                int bit = (readings[i*SENSORS+j] > cut[j]) == (sign[j] > 0);
                history[j] = (history[j] << 1) | bit;
            }
        refresh(history, weights, suspects);
    }
    for (int i = 0; i < n; i++) {
        const double *x = readings + (WARM+i)*SENSORS;
        if (mode == SINGLE) {
            pred[i] = (x[best] > cut[best]) == (sign[best] > 0);
            continue;
        }
        int suspects_sum = 0, group_size = 0;
        double total = 0;
        for (int j = 0; j < SENSORS; j++) {
            int bit = (x[j] > cut[j]) == (sign[j] > 0);
            int v = 2 * bit - 1;
            if (dynamic) {
                total += weights[j] * v;
                if (mode != 0) history[j] = (history[j] << 1) | bit;
                if (mode == REACTIVE) {
                    suspects_sum += suspects[j] * v;
                    group_size += suspects[j];
                }
            } else {
                total += mode == SKILL ? skill[j] * v : v;
            }
        }
        pred[i] = total > 0;
        age++;
        if (mode == REACTIVE) {
            int flag = !group_size || 4 * abs(suspects_sum) < 3 * group_size;
            flags = ((flags << 1) | flag) & 15;
        }
        int trigger = mode == REACTIVE && age >= 6 && __builtin_popcount((unsigned)flags) >= 2;
        if (i+1 < n && ((mode > 0 && age == mode) || trigger)) {
            refresh(history, weights, suspects);
            age = 0;
            flags = 0;
            count++;
        }
    }
    return count;
}
