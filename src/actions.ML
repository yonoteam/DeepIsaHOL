
signature ACTIONS =
sig
  (* Actions.T *)
  type T
  val void: T
  val build: string -> Toplevel.transition -> T
  val text_of: T -> string
  val transition_of: T -> Toplevel.transition
  val kind_of: T -> string
  val pos_of: T -> Position.T
  val on_text: (string -> 'a) -> T -> 'a
  val on_kind: (string -> 'a) -> T -> 'a
  val make: theory -> string -> T list
  val apply_safe: T -> Toplevel.state -> Toplevel.state * (exn * string) option
  val apply: T -> Toplevel.state -> Toplevel.state
   
  (* Data preparation *)
  val begin_thy_from: Toplevel.state -> T list -> theory
  val theory_separators: theory -> (string * (T -> bool) * (T -> bool)) list
  val next_state: T Sections.T -> Toplevel.state -> Toplevel.state
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

val void = Action {text="", transition=Toplevel.ignored Position.none}

fun text_of (Action a) = #text a;

fun transition_of (Action a) = #transition a;

fun kind_of (Action a) = Toplevel.name_of (#transition a);

fun pos_of (Action a) = Toplevel.pos_of (#transition a);

(* on_text: (string -> 'a) -> Actions.T -> 'a *)
fun on_text f a = f (text_of a);

(* on_kind: (string -> 'a) -> Actions.T -> 'a *)
fun on_kind f a = f (kind_of a);

fun build txt tr = Action {text=txt, transition=tr}

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

(* apply_safe: Actions.T -> Toplevel.state -> Toplevel.state * (exn * string) option *)
fun apply_safe (Action a) st = Toplevel.transition true (#transition a) st;

fun apply act st = fst (apply_safe act st);

(* begin_thy_from: Toplevel.state -> Actions.T list -> theory *)
fun begin_thy_from state acts = 
  let
    fun begins_thy a = on_kind Seps.begins_thy a;
  in state
  |> fold apply (Pred.take_until begins_thy acts)
  |> Toplevel.theory_of end;


(** DATA PREPARATION **)

(* theory_separators: theory -> (string * (T -> bool) * (T -> bool)) list *)
fun theory_separators thy =
  let 
    val cmmnds = Get.grouped_commands thy;
    val seps = Seps.for_theory cmmnds
  in map (fn (tag, starts, ends) => (tag, on_kind starts, on_kind ends)) seps end;

(* next_state: Actions.T Sections.T -> Toplevel.state -> Toplevel.state *)
val next_state = Sections.next_state apply

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