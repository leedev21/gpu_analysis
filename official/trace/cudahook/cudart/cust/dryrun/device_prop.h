#pragma once

struct cudaDeviceProp;

namespace hook {
bool QueryDeviceProp(cudaDeviceProp*, unsigned int);
}