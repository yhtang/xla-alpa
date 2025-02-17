#ifndef XLA_SERVICE_GPU_AUTO_SHARDING_H_
#define XLA_SERVICE_GPU_AUTO_SHARDING_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class AutoSharding : public HloModulePass {
 public:
  AutoSharding() = default;
  ~AutoSharding() override = default;
  absl::string_view name() const override { return "auto_sharding"; }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace spmd
}  // namespace xla

#endif // XLA_SERVICE_GPU_AUTO_SHARDING_H_
