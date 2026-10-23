import Std
set_option autoImplicit false

namespace LightNest.Patch

/-- Total fixed-length overwrite. Bounds are enforced by checkedPatch. -/
def overwrite {α : Type} : List α → Nat → α → List α
  | [], _, _ => []
  | _ :: xs, 0, value => value :: xs
  | x :: xs, index + 1, value => x :: overwrite xs index value

def readAt {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, index + 1 => readAt xs index

theorem overwrite_length {α : Type} (xs : List α) (index : Nat) (value : α) :
    (overwrite xs index value).length = xs.length := by
  induction xs generalizing index with
  | nil => simp [overwrite]
  | cons x xs ih => cases index <;> simp [overwrite, ih]

theorem overwrite_reads_back {α : Type} (xs : List α) (index : Nat) (value : α)
    (h : index < xs.length) : readAt (overwrite xs index value) index = some value := by
  induction xs generalizing index with
  | nil => simp at h
  | cons x xs ih =>
    cases index with
    | zero => rfl
    | succ index =>
      simp only [List.length_cons, Nat.succ_lt_succ_iff] at h
      exact ih index h

theorem overwrite_preserves_other {α : Type} (xs : List α) (index other : Nat)
    (value : α) (h : index ≠ other) :
    readAt (overwrite xs index value) other = readAt xs other := by
  induction xs generalizing index other with
  | nil => simp [overwrite, readAt]
  | cons x xs ih =>
    cases index <;> cases other <;> simp_all [overwrite, readAt]

theorem independent_overwrites_commute {α : Type} (xs : List α) (i j : Nat)
    (a b : α) (h : i ≠ j) :
    overwrite (overwrite xs i a) j b = overwrite (overwrite xs j b) i a := by
  induction xs generalizing i j with
  | nil => simp [overwrite]
  | cons x xs ih => cases i <;> cases j <;> simp_all [overwrite]

structure Edit where
  offset : Nat
  before : Nat
  after : Nat
  deriving Repr, DecidableEq

def applyEdits (code : List Nat) : List Edit → List Nat
  | [] => code
  | e :: es => applyEdits (overwrite code e.offset e.after) es

theorem applyEdits_length (code : List Nat) (edits : List Edit) :
    (applyEdits code edits).length = code.length := by
  induction edits generalizing code with
  | nil => rfl
  | cons e es ih =>
    simp only [applyEdits]
    rw [ih, overwrite_length]

theorem applyEdits_preserves_untouched (code : List Nat) (edits : List Edit)
    (index : Nat) (h : ∀ e ∈ edits, e.offset ≠ index) :
    readAt (applyEdits code edits) index = readAt code index := by
  induction edits generalizing code with
  | nil => rfl
  | cons e es ih =>
    have head : e.offset ≠ index := h e (by simp)
    have tail : ∀ a ∈ es, a.offset ≠ index := by
      intro a ha
      exact h a (by simp [ha])
    simp only [applyEdits]
    rw [ih _ tail, overwrite_preserves_other code e.offset index e.after head]

def uniqueOffsets : List Edit → Bool
  | [] => true
  | e :: es => es.all (fun other => e.offset != other.offset) && uniqueOffsets es

def validEdit (code : List Nat) (e : Edit) : Bool :=
  decide (e.before < 256 ∧ e.after < 256) &&
    decide (readAt code e.offset = some e.before) && (e.before != e.after)

/-- Hash equality is an external premise, not a proof of hash collision resistance. -/
def validPatch (code : List Nat) (edits : List Edit) (limit : Nat) (baseMatches : Bool) : Bool :=
  baseMatches && decide (4 ≤ code.length ∧ code.length ≤ 1024 ∧ code.length % 4 = 0) &&
    decide (limit ≤ 8 ∧ edits.length ≤ limit) && code.all (fun b => b < 256) &&
    uniqueOffsets edits && edits.all (validEdit code)

/-- All preconditions are checked against the original, immutable code. -/
def checkedPatch (code : List Nat) (edits : List Edit) (limit : Nat) (baseMatches : Bool) :
    Option (List Nat) :=
  if validPatch code edits limit baseMatches then some (applyEdits code edits) else none

theorem checkedPatch_requires_validity (code out : List Nat) (edits : List Edit)
    (limit : Nat) (baseMatches : Bool)
    (h : checkedPatch code edits limit baseMatches = some out) :
    validPatch code edits limit baseMatches = true := by
  unfold checkedPatch at h
  split at h
  · assumption
  · cases h

theorem checkedPatch_preserves_length (code out : List Nat) (edits : List Edit)
    (limit : Nat) (baseMatches : Bool)
    (h : checkedPatch code edits limit baseMatches = some out) : out.length = code.length := by
  unfold checkedPatch at h
  split at h
  · have he := Option.some.inj h
    rw [← he]
    exact applyEdits_length code edits
  · cases h

theorem checkedPatch_preserves_untouched (code out : List Nat) (edits : List Edit)
    (limit index : Nat) (baseMatches : Bool)
    (h : checkedPatch code edits limit baseMatches = some out)
    (untouched : ∀ e ∈ edits, e.offset ≠ index) : readAt out index = readAt code index := by
  unfold checkedPatch at h
  split at h
  · have he := Option.some.inj h
    rw [← he]
    exact applyEdits_preserves_untouched code edits index untouched
  · cases h

theorem stale_base_rejected (code : List Nat) (edits : List Edit) (limit : Nat) :
    checkedPatch code edits limit false = none := by
  simp [checkedPatch, validPatch]

end LightNest.Patch
