
signature ACTIONS =
sig
  type T
  val void: T
  val text_of: T -> string
  val transition_of: T -> Toplevel.transition
  val kind_of: T -> string
  val pos_of: T -> Position.T
  val make: theory -> string -> T list
  val apply: T -> Toplevel.state -> Toplevel.state * (exn * string) option
  val next_state: T -> Toplevel.state -> Toplevel.state
  val begins_thy: T -> bool
  val begin_thy_from: Toplevel.state -> T list -> theory
  val ends_thy_section: T -> bool
  type section
  val theory_separators: theory -> (string * (T -> bool) * (T -> bool)) list
  val latest_open: section -> section
  val segmentate: (string * (T -> bool) * (T -> bool)) list -> T list -> section
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

(* inspired from Portal-to-Isabelle's parse_text MLFunction2 *)
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


(** KEYWORD PREDICATES **)

(* (string -> 'a) -> Actions.T -> 'a *)
fun on_text f a = f (text_of a);

(* (string -> 'a) -> Actions.T -> 'a *)
fun on_kind f a = f (kind_of a);


(* THEORY *)
val theory_tag = "THEORY"
(* val thy_beginN = "thy_begin" (* i.e. theory *) *)
val theoryN = "theory"
val beginN = "begin"
(* val thy_endN = "thy_end" (* i.e. end *) *)
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
val thy_block_tag = "THY_BLOCK"
val thy_decl_blockN = "thy_decl_block" (* e.g. locale, instantiation *)
val notepadN = "notepad"

(**)
fun starts_notepad a = Pred.starts_with notepadN (text_of a)

(* Action.T -> bool *)
val ends_thy_section = on_kind (curry (op =) endN)


(* THEORY ADDITIONS *)
val lemmasN = "lemmas"
(* 
val thy_defnN = "thy_defn" (* e.g. abbreviation *)
val thy_declN = "thy_decl" (* e.g. no_notation *)
val thy_stmtN = "thy_stmt" (* i.e. axiomatization *)
val thy_loadN = "thy_load" (* e.g. ML_file *)
val diagN = "diag" (* e.g. term *) 
*)


(* PROOF *)
val proof_tag = "PROOF"
val goalN = "goal"
(* 
val thy_goalN = "thy_goal" (* e.g. instance *)
val thy_goal_stmtN = "thy_goal_stmt" (* e.g. lemma *)
val thy_goal_defnN = "thy_goal_defn" (* e.g. termination *)
*)

(* APPLY-STYLE PROOFS *)
val qedN = "qed" (* i.e. by *)
(* 
val prf_scriptN = "prf_script" (* e.g. apply *)
val prf_script_goalN = "prf_script_goal" (* i.e. subgoal *)
val prf_declN = "prf_decl" (* e.g. unfolding *)
val qed_scriptN = "qed_script" (* i.e. done *)

val qed_globalN = "qed_global" (* i.e. oops *) 
*)

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

(*  theory -> (string * (T -> bool) * (T -> bool)) list *)
fun theory_separators thy =
  let 
    val cmmnds = Get.grouped_commands thy;
    fun in_group_cmmnds P a = 
      let
        val kwds = maps snd (filter (Pred.on_fst P) cmmnds);
      in member (op =) kwds (kind_of a) end;
    
    val thy_seps = (theory_tag, begins_thy, ends_thy_section);

    val starts_thy_block = in_group_cmmnds (fn grp_name => grp_name = thy_decl_blockN);    
    val thy_block_seps = (thy_block_tag, starts_thy_block, ends_thy_section 
          orf (starts_thy_block andf (Pred.neg starts_notepad)));
    
    fun starts_proof a = on_kind (fn txt => not (txt = lemmasN)) a
      andalso in_group_cmmnds (fn str => Pred.contains goalN str) a
    fun ends_proof a = in_group_cmmnds (fn str => Pred.contains qedN str) a
    val thy_proof_seps = (proof_tag, starts_proof, ends_proof); 

  in [thy_proof_seps, thy_block_seps, thy_seps] end;


(** SECTIONS **)

datatype section = 
    Empty
  | Single of T
  | Block of string * bool * (section list);


fun is_closed (Block (_, closed, sects)) = closed andalso forall is_closed sects
  | is_closed _ = true;


local 

(* Actions.section -> Actions.section *)
fun reverse (Block (tag, closed, sects)) = Block (tag, closed, rev (map reverse sects))
  | reverse sect = sect;

(* Actions.section -> Actions.section *)
fun top_last_tag Empty = NONE
  | top_last_tag (Single _) = NONE
  | top_last_tag (Block (tag, _, _)) = SOME tag

(* Actions.section -> Actions.section *)
fun close_latest (Block (tag, closed, sects)) = 
      if null sects then raise Fail ("empty block at" ^ tag)
      else if not closed then
        if forall is_closed sects then Block (tag, true, sects)
        else Block (tag, closed, close_latest (hd sects) :: (tl sects))
      else raise Fail ("tried to close closed block" ^ tag)
  |  close_latest _ = raise Fail ("tried to close closed nonblock");

(* Actions.T * string list * string list -> Actions.section -> Actions.section *)
fun add_new_section (a, starts, ends) (Empty) = 
      if not (null starts) then Block (hd starts, false, [Single a])
      else if not (null ends) then raise Fail ("Unbalanced blocks on " ^ text_of a)
      else Block ("OTHER", false, [Single a])
  | add_new_section _ (Single a) = 
      raise Fail ("Wrong call add_new_section' on Single of" ^ text_of a)
  | add_new_section (a, starts, ends) (Block (block_tag, false, sects)) =
      if null sects then 
        raise Fail ("Nil input in Block before " ^ text_of a)
      else if is_closed (hd sects) then
        if not (null starts) then Block (block_tag, false, (Block (hd starts, false, [Single a])) :: sects)
        else if not (null ends) then
          if member (op =) ends block_tag then Block (block_tag, true, Single a :: sects)
          else raise Fail ("Closing " ^ block_tag ^ " block with " ^ hd ends ^ " at " ^ text_of a)
        else Block (block_tag, false, Single a :: sects)
      else
        if not (null starts) andalso not (null ends) then
          if hd ends = the (top_last_tag (hd sects)) then 
            add_new_section (a, starts, ends) (Block (block_tag, false, (close_latest (hd sects)) :: tl sects))
          else 
            Block (block_tag, false, (add_new_section (a, starts, ends) (hd sects)) :: tl sects)
        else
          Block (block_tag, false, (add_new_section (a, starts, ends) (hd sects)) :: tl sects)
  | add_new_section (a, _, _) (Block (_, true, sects)) = 
      if null sects then raise Fail ("Null sections for input Block") 
      else raise Fail ("unbalanced blocks at " ^ text_of a);

fun get_separators (a:T) [] = (a, [], [])
  | get_separators a ((tag, starts, ends) :: tses) =
      if starts a andalso ends a then 
        (a, tag :: (#2 (get_separators a tses)), tag :: (#3 (get_separators a tses)))
      else if starts a then 
        (a, tag :: (#2 (get_separators a tses)), #3 (get_separators a tses))
      else if ends a then 
        (a, #2 (get_separators a tses), tag :: (#3 (get_separators a tses)))
      else get_separators a tses;

in

(* Actions.section -> Actions.section *)
fun latest_open (Block (tag, closed, sects)) = 
      if null sects then raise Fail ("empty block at" ^ tag)
      else if forall is_closed sects then
        if not closed then Block (tag, closed, sects) else Empty
      else Data_Ops.last (map latest_open sects)
  |  latest_open _ = Empty;

(* (string * (Actions.T -> bool) * (Actions.T -> bool)) list -> Actions.T list -> Actions.section *)
fun segmentate tags_starts_ends acts =
  let
    val rev_result = 
      let
        fun iter (a:T) sect = add_new_section (get_separators a tags_starts_ends) sect
      in fold iter acts Empty end;
  in reverse rev_result end;

end;

end;