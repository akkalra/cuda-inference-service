#include <torch/extension.h>

void fused_rmsnorm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_rmsnorm", &fused_rmsnorm, "Fused RMSNorm (CUDA)");
}
