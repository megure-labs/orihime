// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>

#ifdef USE_TORCH_LIBRARY
TORCH_LIBRARY_FRAGMENT(orihime, m) {
  m.def("grid_forward_messages(Tensor inputs, Tensor lengths, Tensor topology, "
        "int algorithm, float parameter0, float parameter1, float parameter2, "
        "float temperature, int bandwidth) -> (Tensor, Tensor)");
}
#endif
