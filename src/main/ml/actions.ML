(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Representation of Isabelle's transitions as user actions.
*)

signature ACTIONS =
sig
  (* Actions.T *)
  type T
  val void: T
  val text_of: T -> string
  val transition_of: T -> Toplevel.transition
  val kind_of: T -> string
  val pos_of: T -> Position.T
  val on_text: (string -> 'a) -> T -> 'a
  val on_kind: (string -> 'a) -> T -> 'a
  val make: theory -> string -> T list
  val make_one: theory -> string -> T
  val make': theory -> string -> T list

  (* Application *)
  val apply_safe: T -> Toplevel.state -> Toplevel.state * (exn * string) option
  val apply: T -> Toplevel.state -> Toplevel.state
  val next_state: T Sections.T -> Toplevel.state -> Toplevel.state
  val apply_all: T list -> Toplevel.state -> (T * Toplevel.state * (exn * string) option) list
  val apply_all': T list -> Toplevel.state -> (Toplevel.state * T * (exn * string) option) list
  val apply_while: (Toplevel.state * T * (exn * string) option -> bool) -> T list  -> Toplevel.state
    -> (Toplevel.state * T * (exn * string) option) list * T list
  val timed_apply_all: {timeout_in_secs: int} -> T list -> Toplevel.state -> (T * Toplevel.state * (exn * string) option) list

  (* Data preparation *)
  val theory_separators: theory -> (string * (T -> bool) * (T -> bool)) list
  val segmentate: Toplevel.state -> (string * (T -> bool) * (T -> bool)) list -> T list 
    -> (Toplevel.state * T) Sections.T
  val segmentate_triplets: Toplevel.state -> (string * (T -> bool) * (T -> bool)) list -> T list 
    -> (Toplevel.state * T * Toplevel.state) Sections.T
  val segmentate_stateless: theory -> T list -> T Sections.T
end;


structure Actions: ACTIONS =
struct


(** ACTIONS **)

datatype T = Action of {
  text: string,
  transition: Toplevel.transition}

fun build txt tr = Action {text=txt, transition=tr};

val void = build "" (Toplevel.ignored Position.none);

fun text_of (Action a) = #text a;

fun transition_of (Action a) = #transition a;

fun kind_of (Action a) = Toplevel.name_of (#transition a);

fun pos_of (Action a) = Toplevel.pos_of (#transition a);

(* on_text: (string -> 'a) -> Actions.T -> 'a *)
fun on_text f act = f (text_of act);

(* on_kind: (string -> 'a) -> Actions.T -> 'a *)
fun on_kind f act = f (kind_of act);

(* inspired by Portal-to-Isabelle's parse_text MLFunction2 *)
(* make: theory -> string -> Actions.T list *)
fun make thy0 thy_text =
  let
    val trs = Get.transitions thy0 Position.start thy_text;
    fun dist tr1 tr2 = Option.valOf (Position.distance_of (Toplevel.pos_of tr1, Toplevel.pos_of tr2));
    fun record_acts syms [tr] = [build (implode syms) tr]
      | record_acts _ [] = []
      | record_acts syms (tr1 :: tr2 :: trs) = 
          let 
           val (syms1, syms2) = Library.chop (dist tr1 tr2) syms;
           val act = build (implode syms1) tr1;
          in act :: record_acts syms2 (tr2 :: trs) end;
  in record_acts (Symbol.explode thy_text) trs end;

(* make_one: theory -> string -> Actions.T *)
fun make_one thy0 thy_text = hd (make thy0 thy_text)

(* make': theory -> string -> Actions.T list *)
fun make' thy0 thy_text =
  let
    val does_something = on_kind (
      Pred.is_not "<ignored>" 
      andf Pred.neg (Seps.is_document_step (Get.grouped_commands thy0)))
  in void :: (filter does_something (make thy0 thy_text)) end;


(** APPLICATION **)

(* apply_safe: Actions.T -> Toplevel.state -> Toplevel.state * (exn * string) option *)
fun apply_safe (Action a) st = Toplevel.transition true (#transition a) st;

(* apply: Actions.T -> Toplevel.state -> Toplevel.state *)
fun apply act st = fst (apply_safe act st);

(* next_state: Actions.T Sections.T -> Toplevel.state -> Toplevel.state *)
val next_state = Sections.next_state apply;

(* apply_all: Actions.T list -> Toplevel.state 
  -> (Actions.T * Toplevel.state * (exn * string) option) list *)
fun apply_all [] _ = []
  | apply_all acts st =
  let
    fun do_next act' (_, st, _) =
      let
        val (st', err') = apply_safe act' st
      in (act', st', err') end;
  in Ops.log_fold do_next (void, st, NONE) acts end;

(* apply_all': Actions.T list -> Toplevel.state 
  -> (Toplevel.state * Actions.T * (exn * string) option) list *)
fun apply_all' [] st = [(st, void, NONE)]
  | apply_all' acts st =
  let
    fun do_next act (st, so_far) =
      let
        val (st', err') = apply_safe act st;
      in (st', (st, act, err') :: so_far) end;
    val (_, rev_result) = fold do_next acts (st, [])
  in rev rev_result end;

(* apply_while: (Toplevel.state * Actions.T * (exn * string) option -> bool) 
  -> Actions.T list -> Toplevel.state 
  -> (Toplevel.state * Actions.T * (exn * string) option) list * Actions.T list *)
fun apply_while P acts st =
  let
    fun do_next act' (st, _, _) =
      let
        val (st', err') = apply_safe act' st;
      in (st', act', err') end;
  in Pred.log_fold P do_next acts (st, void, NONE) end;

(* timed_apply_all: {timeout_in_secs: int} 
  -> Actions.T list -> Toplevel.state 
  -> (Actions.T * Toplevel.state * (exn * string) option) list *)
  fun timed_apply_all {timeout_in_secs=t} acts st =
  let
    val t' = Time.fromSeconds t
    val danger_methods = ["smt", "metis", "simp", "blast", "auto"]
    val is_danger_method = Pred.disjunct (map Pred.contains danger_methods)
    val might_timeout = on_kind (Pred.is "apply") andf on_text is_danger_method;
    fun timeout_apply act st = case Ops.apply_with_timeout t' (apply_safe act) st of
        Exn.Exn exn => (st, SOME (exn, "TIMEOUT"))
      | Exn.Res result => result
    fun do_next act' (_, st, _) =
    let
      val (st', err') = if might_timeout act'
        then timeout_apply act' st
        else apply_safe act' st
    in (act', st', err') end;
  in Ops.log_fold do_next (void, st, NONE) acts end;


(** DATA PREPARATION **)

(* theory_separators: theory -> (string * (T -> bool) * (T -> bool)) list *)
fun theory_separators thy =
  let 
    val cmmnds = Get.grouped_commands thy;
    val seps = Seps.make cmmnds
  in map (fn (tag, starts, ends) => (tag, on_kind starts, on_kind ends)) seps end;

(* segmentate: Toplevel.state -> (string * (Actions.T -> bool) * (Actions.T -> bool)) list 
  -> Actions.T list -> (Toplevel.state * Actions.T) Sections.T *)
fun segmentate state0 separators acts =
  let
    val init = if null acts then (state0, void) else (state0, hd acts);
    fun iter a' (s, a) = (apply a s, a');
    val steps = Ops.log_fold iter init (tl acts)
    fun lift f (_, a) = f a;
    fun lifting (tag, is_start, is_end) = (tag, lift is_start, lift is_end);
  in Sections.raw_segmentate (lift text_of) (map lifting separators) steps end;

(* segmentate_triplets: Toplevel.state -> (string * (Actions.T -> bool) * (Actions.T -> bool)) list 
  -> Actions.T list -> (Toplevel.state * Actions.T * Toplevel.state) Sections.T *)
fun segmentate_triplets state0 separators acts =
  let
    val init = (state0, void, state0);
    fun iter a (_, _, s) = (s, a, apply a s);
    val triplets = Ops.log_fold iter init acts
    fun lift f (_, a, _) = f a;
    fun lifting (tag, is_start, is_end) = (tag, lift is_start, lift is_end);
  in Sections.raw_segmentate (lift text_of) (map lifting separators) triplets end;

(* segmentate_stateless: theory -> Actions.T list -> Actions.T Sections.T *)
fun segmentate_stateless thy acts =
  let
    val separators = theory_separators thy
  in Sections.raw_segmentate (text_of) separators acts end;

end;