---- MODULE ModelPromotionLanes ----
EXTENDS FiniteSets

(***************************************************************************)
(* Accepted/candidate/rejected model discipline and replay-lane separation. *)
(*                                                                         *)
(* Intent: candidate or rejected checkpoints must not generate future       *)
(* self-play data; clean zero replay must never reference a supervised      *)
(* source model; supervised self-play must always reference an accepted     *)
(* source model.                                                           *)
(***************************************************************************)

CONSTANTS Models

VARIABLES modelState, zeroSources, supSpSources, crossplaySources

ModelStates == {"candidate", "evaluating", "accepted", "rejected"}

vars == << modelState, zeroSources, supSpSources, crossplaySources >>

Init ==
  /\ modelState = [m \in Models |-> "candidate"]
  /\ zeroSources = {}
  /\ supSpSources = {}
  /\ crossplaySources = {}

BeginEval(m) ==
  /\ modelState[m] = "candidate"
  /\ modelState' = [modelState EXCEPT ![m] = "evaluating"]
  /\ UNCHANGED << zeroSources, supSpSources, crossplaySources >>

Accept(m) ==
  /\ modelState[m] = "evaluating"
  /\ modelState' = [modelState EXCEPT ![m] = "accepted"]
  /\ UNCHANGED << zeroSources, supSpSources, crossplaySources >>

Reject(m) ==
  /\ modelState[m] \in {"candidate", "evaluating"}
  /\ modelState' = [modelState EXCEPT ![m] = "rejected"]
  /\ UNCHANGED << zeroSources, supSpSources, crossplaySources >>

GenerateZero ==
  /\ zeroSources' = zeroSources \cup {"none"}
  /\ UNCHANGED << modelState, supSpSources, crossplaySources >>

GenerateSupSp(m) ==
  /\ modelState[m] = "accepted"
  /\ supSpSources' = supSpSources \cup {m}
  /\ UNCHANGED << modelState, zeroSources, crossplaySources >>

GenerateCrossplay(m) ==
  /\ modelState[m] = "accepted"
  /\ crossplaySources' = crossplaySources \cup {m}
  /\ UNCHANGED << modelState, zeroSources, supSpSources >>

Next ==
  \/ \E m \in Models: BeginEval(m)
  \/ \E m \in Models: Accept(m)
  \/ \E m \in Models: Reject(m)
  \/ GenerateZero
  \/ \E m \in Models: GenerateSupSp(m)
  \/ \E m \in Models: GenerateCrossplay(m)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ modelState \in [Models -> ModelStates]
  /\ zeroSources \subseteq {"none"}
  /\ supSpSources \subseteq Models
  /\ crossplaySources \subseteq Models

SupSpSourcesAreAccepted ==
  \A m \in supSpSources: modelState[m] = "accepted"

CrossplaySourcesAreAccepted ==
  \A m \in crossplaySources: modelState[m] = "accepted"

ZeroHasNoSourceModel ==
  zeroSources \subseteq {"none"}

THEOREM Spec => []TypeOK
THEOREM Spec => []SupSpSourcesAreAccepted
THEOREM Spec => []CrossplaySourcesAreAccepted
THEOREM Spec => []ZeroHasNoSourceModel

====
