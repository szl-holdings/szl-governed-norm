{
  description = "Flake for the SZL Holdings szl_governed_norm universal kernel";
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
