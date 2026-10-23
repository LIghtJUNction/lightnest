import Std

/- Mechanism specifications, not a proof of the Python search or RV32 VM.
   Retaining an incomplete candidate cannot authorize committing it. -/
namespace LightNest.RepairMemory

def passes : List Bool → Bool
  | [] => true
  | b :: rest => b && passes rest

def checkedCommit {α : Type} (candidate : α) (checks : List Bool) : Option α :=
  if passes checks = true then some candidate else none

def extendSearch {α : Type} (existing : Option α) (fallback : Unit → Option α) : Option α :=
  match existing with
  | some result => some result
  | none => fallback ()

theorem prefix_is_not_completion : passes [true, false] = false := by decide

theorem commit_requires_all {α : Type} (candidate : α) (checks : List Bool)
    (h : checkedCommit candidate checks = some candidate) : passes checks = true := by
  by_cases hp : passes checks = true
  · exact hp
  · simp [checkedCommit, hp] at h

theorem incomplete_is_rejected {α : Type} (candidate : α) (checks : List Bool)
    (h : passes checks = false) : checkedCommit candidate checks = none := by
  simp [checkedCommit, h]

theorem passes_append (left right : List Bool) :
    passes (left ++ right) = (passes left && passes right) := by
  induction left with
  | nil => simp [passes]
  | cons b rest ih =>
    cases b <;> simp [passes, ih]

theorem additional_checks_keep_old (old added : List Bool)
    (h : passes (old ++ added) = true) : passes old = true := by
  rw [passes_append] at h
  cases hold : passes old <;> simp_all

theorem existing_result_preserved {α : Type} (result : α) (fallback : Unit → Option α) :
    extendSearch (some result) fallback = some result := by rfl

theorem attempt_fits_budget (used limit : Nat) (h : used < limit) :
    used + 1 ≤ limit := by
  exact Nat.succ_le_of_lt h

end LightNest.RepairMemory
