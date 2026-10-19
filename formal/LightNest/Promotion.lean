import Std
set_option autoImplicit false

namespace LightNest.Promotion

/-- The Python adapter first aligns unique task IDs and rejects unequal ID sets. -/
structure Comparison where
  parentSuccess : Bool
  candidateSuccess : Bool
  candidateFalseEdit : Bool
  deriving Repr, DecidableEq

def acceptable (row : Comparison) : Bool :=
  (!row.parentSuccess || row.candidateSuccess) && !row.candidateFalseEdit

/-- A nonempty, already aligned evaluation set. This is a safety gate, not full promotion. -/
def check (rows : List Comparison) : Bool := !rows.isEmpty && rows.all acceptable

theorem accepted_row (rows : List Comparison) (row : Comparison)
    (h : check rows = true) (member : row ∈ rows) : acceptable row = true := by
  have allRows : rows.all acceptable = true := by
    unfold check at h
    cases he : rows.isEmpty <;> simp_all
  exact (List.all_eq_true.mp allRows) row member

theorem no_old_success_lost (rows : List Comparison) (row : Comparison)
    (h : check rows = true) (member : row ∈ rows) (old : row.parentSuccess = true) :
    row.candidateSuccess = true := by
  have ok := accepted_row rows row h member
  cases row with
  | mk p c f => cases p <;> cases c <;> cases f <;> simp_all [acceptable]

theorem no_false_edits (rows : List Comparison) (row : Comparison)
    (h : check rows = true) (member : row ∈ rows) : row.candidateFalseEdit = false := by
  have ok := accepted_row rows row h member
  cases row with
  | mk p c f => cases p <;> cases c <;> cases f <;> simp_all [acceptable]

theorem empty_evidence_rejected : check [] = false := rfl

/-- Preservation composes across generations on the SAME declared task domain. -/
def Preserves {ι : Type} (parent candidate : ι → Bool) : Prop :=
  ∀ i, parent i = true → candidate i = true

theorem preserves_trans {ι : Type} (a b c : ι → Bool)
    (ab : Preserves a b) (bc : Preserves b c) : Preserves a c := by
  intro i hi
  exact bc i (ab i hi)

/-- A rejected proposal does not replace the incumbent. -/
def choose {α : Type} (approved : Bool) (parent candidate : α) : α :=
  if approved then candidate else parent

theorem rejected_keeps_parent {α : Type} (parent candidate : α) :
    choose false parent candidate = parent := rfl

theorem choose_preserves_invariant {α : Type} (P : α → Prop) (approved : Bool)
    (parent candidate : α) (hp : P parent) (hc : approved = true → P candidate) :
    P (choose approved parent candidate) := by
  cases approved <;> simp_all [choose]

end LightNest.Promotion
