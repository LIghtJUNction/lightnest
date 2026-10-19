import Lean
import LightNest

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let mut count := 0
  for (name, info) in env.constants.toList do
    if name.toString.startsWith "LightNest." then
      match info with
      | .axiomInfo _ => throwError "unapproved project axiom: {name}"
      | .thmInfo _ => count := count + 1
      | _ => pure ()
      for ax in (← Lean.collectAxioms name) do
        unless allowed.contains ax do
          throwError "unapproved axiom {ax} in {name}"
  if count < 20 then
    throwError "proof inventory unexpectedly small: {count}"
  logInfo m!"LightNest axiom audit passed: {count} theorems"
