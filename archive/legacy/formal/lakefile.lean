import Lake
open Lake DSL

package lightnest_formal where
  moreLeanArgs := #["-DwarningAsError=true"]

@[default_target]
lean_lib LightNest
