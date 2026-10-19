import Std
set_option autoImplicit false

namespace LightNest.Compute

/-- One unchanged, deterministic transition, with the same frozen inputs. -/
def iterate {α : Type} (step : α → α) : Nat → α → α
  | 0, state => state
  | n + 1, state => iterate step n (step state)

theorem split_unroll {α : Type} (step : α → α) (first rest : Nat) (state : α) :
    iterate step (first + rest) state = iterate step rest (iterate step first state) := by
  induction first generalizing state with
  | zero => simp [iterate]
  | succ first ih => simpa [Nat.succ_add, iterate] using ih (step state)

theorem rgb_4_plus_60 {α : Type} (step : α → α) (state : α) :
    iterate step 64 state = iterate step 60 (iterate step 4 state) :=
  split_unroll step 4 60 state

/-- Early output is sound IF future output invariance has been established. -/
def OutputStable {α β : Type} (step : α → α) (decode : α → β) (state : α) : Prop :=
  ∀ rounds : Nat, decode (iterate step rounds state) = decode state

theorem stable_exit_preserves_output {α β : Type} (step : α → α) (decode : α → β)
    (state : α) (spent remaining : Nat)
    (h : OutputStable step decode (iterate step spent state)) :
    decode (iterate step (spent + remaining) state) = decode (iterate step spent state) := by
  rw [split_unroll]
  exact h remaining

/-- Three identical observed actions do not establish OutputStable. -/
def delayedOutput (state : Nat) : Nat := if state < 3 then 0 else 1

theorem repeated_actions_are_not_a_certificate :
    delayedOutput 0 = delayedOutput 1 ∧ delayedOutput 1 = delayedOutput 2 ∧
      delayedOutput 2 ≠ delayedOutput (iterate Nat.succ 3 0) := by decide

inductive Mode where
  | think
  | emit
  deriving Repr, DecidableEq, BEq

/-- Count until the first EMIT, bounded by fuel even if the policy never emits. -/
def stepsUsed : Nat → List Mode → Nat
  | 0, _ => 0
  | _ + 1, [] => 0
  | _ + 1, Mode.emit :: _ => 1
  | fuel + 1, Mode.think :: rest => 1 + stepsUsed fuel rest

theorem stepsUsed_le_fuel (fuel : Nat) (choices : List Mode) : stepsUsed fuel choices ≤ fuel := by
  induction fuel generalizing choices with
  | zero => simp [stepsUsed]
  | succ fuel ih =>
    cases choices with
    | nil => simp [stepsUsed]
    | cons choice rest =>
      cases choice with
      | emit => simp [stepsUsed]
      | think =>
        simp only [stepsUsed]
        have h := ih rest
        omega

theorem emit_ignores_suffix (fuel : Nat) (rest : List Mode) :
    stepsUsed (fuel + 1) (Mode.emit :: rest) = 1 := rfl

/-- Check an absolute-depth schedule; subtraction cannot conceal a backward step. -/
def scheduleCost (previous maximum : Nat) : List Nat → Option Nat
  | [] => if previous = maximum then some 0 else none
  | depth :: rest =>
      if previous < depth ∧ depth ≤ maximum then
        (scheduleCost depth maximum rest).map (fun cost => depth - previous + cost)
      else none

theorem hybrid_schedule_cost : scheduleCost 0 64 [4, 8, 16, 32, 64] = some 64 := by decide

theorem backward_schedule_rejected : scheduleCost 0 64 [8, 4, 64] = none := by decide

/-- Exact integer residual model only; no claim about NaNs or floating point. -/
def residual (state delta gate : Int) : Int := state + gate * delta

theorem zero_residual_preserves (state delta : Int) : residual state delta 0 = state := by
  simp [residual]

end LightNest.Compute
