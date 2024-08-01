
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
  val apply: T -> Toplevel.state -> Toplevel.state * (exn * string) option
  
  (* Separators *)
  val begins_thy: T -> bool
  val begin_thy_from: Toplevel.state -> T list -> theory
  val ends_thy_section: T -> bool
  val theory_separators: theory -> (string * (T -> bool) * (T -> bool)) list
  
  (* Data preparation *)
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

(* inspired by Portal-to-Isabelle's parse_text MLFunction2 *)
(* theory -> string -> Actions.T list *)
fun make thy0 thy_text =
  let
    val trs = Get.transitions thy0 Position.start thy_text;
      fun dist tr1 tr2 = Option.valOf (Position.distance_of (Toplevel.pos_of tr1, Toplevel.pos_of tr2));
      fun make_act txt tr = Action {text=txt, transition=tr}
      fun record_acts syms [tr] = [make_act (implode syms) tr]
        | record_acts _ [] = []
        | record_acts syms (tr1 :: tr2 :: trs) = 
            let 
             val (syms1, syms2) = Library.chop (dist tr1 tr2) syms;
             val act = make_act (implode syms1) tr1;
            in act :: record_acts syms2 (tr2 :: trs) end;
  in record_acts (Symbol.explode thy_text) trs end;

(* Actions.T -> Toplevel.state -> Toplevel.state * (exn * string) option *)
fun apply (Action a) st = Toplevel.transition true (#transition a) st;

(* Actions.T -> Toplevel.state -> Toplevel.state *)
fun next_state a st = fst (apply a st)



(** SEPARATORS **)

(* (string -> 'a) -> Actions.T -> 'a *)
fun on_text f a = f (text_of a);

(* (string -> 'a) -> Actions.T -> 'a *)
fun on_kind f a = f (kind_of a);


(* THEORY *)
(* val thy_beginN = "thy_begin" (* i.e. theory *) *)
(* val thy_endN = "thy_end" (* i.e. end *) *)
val theoryN = "theory"
val beginN = "begin"
val endN = "end"

(* Action.T -> bool *)
val begins_thy = Pred.conjunct [
  on_text (Pred.starts_with theoryN), 
  on_text (Pred.contains beginN), 
  on_kind (curry (op =) theoryN)];

(* Toplevel.state -> Actions.T list -> theory *)
fun begin_thy_from state acts = state
  |> fold next_state (Pred.take_until (begins_thy) acts)
  |> Toplevel.theory_of;


(* THY_BLOCK *)
val thy_decl_blockN = "thy_decl_block" (* e.g. locale, instantiation *)
val notepadN = "notepad"

(* Action.T -> bool *)
fun starts_notepad a = Pred.starts_with notepadN (text_of a)

(* Action.T -> bool *)
val ends_thy_section = on_kind (curry (op =) endN)


(* THEORY ADDITIONS *)
(* 
val thy_defnN = "thy_defn" (* e.g. abbreviation *)
val thy_declN = "thy_decl" (* e.g. no_notation *)
val thy_stmtN = "thy_stmt" (* i.e. axiomatization *)
val thy_loadN = "thy_load" (* e.g. ML_file *)
val diagN = "diag" (* e.g. term *) 
*)
val lemmasN = "lemmas"


(* PROOF *)
(* 
val thy_goalN = "thy_goal" (* e.g. instance *)
val thy_goal_stmtN = "thy_goal_stmt" (* e.g. lemma *)
val thy_goal_defnN = "thy_goal_defn" (* e.g. termination *)
*)
val goalN = "goal"


(* APPLY-STYLE PROOFS *)
(* 
val prf_scriptN = "prf_script" (* e.g. apply *)
val prf_script_goalN = "prf_script_goal" (* i.e. subgoal *)
val prf_declN = "prf_decl" (* e.g. unfolding *)
val qed_scriptN = "qed_script" (* i.e. done *)

val qed_globalN = "qed_global" (* i.e. oops *) 
*)
val qedN = "qed" (* i.e. by *)

(* ISAR-STYLE PROOFS *)
(* 
val prf_blockN = "prf_block" (* i.e. proof *)
val prf_asmN = "prf_asm" (* e.g. define *)
val prf_goalN = "prf_goal" (* e.g. have *)
val prf_closeN = "prf_close" (* i.e. { *)
val prf_openN = "prf_open" (* i.e. } *)
val next_blockN = "next_block" (* i.e. next *)
val prf_chainN = "prf_chain" (* e.g. ultimately *)
val prf_asm_goalN = "prf_asm_goal" (* i.e. show, obtain and thus *)
val qed_blockN = "qed_block" (* i.e. qed *) 
*)


(* DOCUMENT PREPARATION *)
(* 
val document_headingN = "document_heading" (* e.g. section *)
val document_bodyN = "document_body" (* e.g. text *)
val document_rawN = "document_raw" (* i.e. text_raw *) 
*)



(** DATA PREPARATION **)

(*  theory -> (string * (T -> bool) * (T -> bool)) list *)
fun theory_separators thy =
  let 
    val cmmnds = Get.grouped_commands thy;
    fun in_group_cmmnds P a = 
      let
        val kwds = maps snd (filter (Pred.on_fst P) cmmnds);
      in member (op =) kwds (kind_of a) end;
    
    val thy_seps = (Sections.theory_tag, begins_thy, ends_thy_section);

    val starts_thy_block = in_group_cmmnds (fn grp_name => grp_name = thy_decl_blockN);    
    val thy_block_seps = (Sections.thy_block_tag, starts_thy_block, ends_thy_section 
          orf (starts_thy_block andf (Pred.neg starts_notepad)));
    
    fun starts_proof a = on_kind (fn txt => not (txt = lemmasN)) a
      andalso in_group_cmmnds (fn str => Pred.contains goalN str) a
    fun ends_proof a = in_group_cmmnds (fn str => Pred.contains qedN str) a
    val thy_proof_seps = (Sections.proof_tag, starts_proof, ends_proof); 

  in [thy_proof_seps, thy_block_seps, thy_seps] end;

(*  Actions.section -> Toplevel.state -> Toplevel.state *)
val next_state = Sections.next_state (fn a => fn s => fst (apply a s))

(* segmentate: Toplevel.state -> (string * (Actions.T -> bool) * (Actions.T -> bool)) list 
  -> Actions.T list -> (Toplevel.state * Actions.T) Sections.T *)
fun segmentate state0 separators acts =
  let
    val init = if null acts then (state0, void) else (state0, hd acts);
    fun iter a' (s, a) = (fst (apply a s), a');
    val steps = Data_Ops.log_fold iter init (tl acts)
    fun lift f (_, a) = f a;
    fun lifting (tag, is_start, is_end) = (tag, lift is_start, lift is_end);
  in Sections.raw_segmentate (lift text_of) (map lifting separators) steps end;

(* segmentate_triplets: Toplevel.state -> (string * (Actions.T -> bool) * (Actions.T -> bool)) list 
  -> Actions.T list -> (Toplevel.state * Actions.T * Toplevel.state) Sections.T *)
fun segmentate_triplets state0 separators acts =
  let
    val init = (state0, void, state0);
    fun iter a (_, _, s) = (s, a, fst (apply a s));
    val triplets = Data_Ops.log_fold iter init acts
    fun lift f (_, a, _) = f a;
    fun lifting (tag, is_start, is_end) = (tag, lift is_start, lift is_end);
  in Sections.raw_segmentate (lift text_of) (map lifting separators) triplets end;

(* segmentate_stateless: theory -> Actions.T list -> Actions.T Sections.T *)
fun segmentate_stateless thy acts =
  let
    val separators = theory_separators thy
  in Sections.raw_segmentate (text_of) separators acts end;

end;