/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/optimization_barrier_expander.h"

namespace xla {

StatusOr<bool> OptimizationBarrierExpander::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> barriers;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    bool modified = false;
    for (HloInstruction* inst : computation->instructions()) {
      // Modified by Alpa: add pipeline marker option
      if (inst->IsCustomCall("pipeline_marker") || inst->opcode() == HloOpcode::kOptimizationBarrier) {
        barriers.push_back(inst);
        modified = true;
      }
    }
  }

  // Modified by Alpa: remove the module->has_schedule() branch;
  // @yhtang: may not be needed anymore

  for (HloInstruction* inst : barriers) {
    HloInstruction* arg = inst->mutable_operand(0);
    TF_RETURN_IF_ERROR(arg->CopyAllControlDepsFrom(inst));

    TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(arg));
    TF_RETURN_IF_ERROR(inst->DropAllControlDeps());

    TF_RETURN_IF_ERROR(inst->parent()->RemoveInstruction(inst));
  }

  return !barriers.empty();
}

}  // namespace xla
