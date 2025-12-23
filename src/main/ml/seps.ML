(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Utility for finding the keywords separating beginnings and ends of code-block in an Isabelle .thy file.
*)

signature SEPS =
sig
  (* Separators *)
  val begins_thy: string -> bool
  val ends_thy_section: string -> bool
  val is_in_group_where: (string * string list) list -> (string -> bool) -> (string -> bool)
  val starts_notepad: string -> bool
  val starts_thy_block: (string * string list) list -> string -> bool
  val ends_thy_block: (string * string list) list -> string -> bool
  val starts_proof: (string * string list) list -> string -> bool
  val declares_proof: (string * string list) list -> string -> bool
  val ends_proof: (string * string list) list -> string -> bool
  val declares_local_goal: (string * string list) list -> string -> bool
  val is_apply_step: (string * string list) list -> string -> bool
  val is_isar_step: (string * string list) list -> string -> bool
  val is_document_step: (string * string list) list -> string -> bool
end;


structure Seps: SEPS =
struct


(** SEPARATORS **)


(* THEORY *)
(* val thy_beginN = "thy_begin" (* i.e. theory *) *)
(* val thy_endN = "thy_end" (* i.e. end *) *)
val theoryN = "theory"
val endN = "end"

val begins_thy = Pred.is theoryN

val ends_thy_section = Pred.is endN


(* THY_BLOCK *)
val thy_decl_blockN = "thy_decl_block" (* e.g. locale, instantiation *)
val notepadN = "notepad"

(* is_in_group_where: commands -> (string -> bool) -> string -> bool *)
fun is_in_group_where grpd_cmmnds P command = 
  let
    val kwds = maps snd (filter (Pred.on_fst P) grpd_cmmnds);
  in member (op =) kwds command end;

val starts_notepad = Pred.starts_with notepadN

fun starts_thy_block grpd_cmmnds = is_in_group_where grpd_cmmnds (Pred.is thy_decl_blockN);

fun ends_thy_block grpd_cmmnds = ends_thy_section
  orf (starts_thy_block grpd_cmmnds andf (Pred.neg starts_notepad))

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
val thy_goalN = "thy_goal" (* e.g. instance *)
val thy_goal_stmtN = "thy_goal_stmt" (* e.g. lemma *)
val thy_goal_defnN = "thy_goal_defn" (* e.g. termination *)
val goalN = "goal"

fun declares_proof grpd_cmmnds = is_in_group_where grpd_cmmnds (
  Pred.is thy_goalN 
  orf Pred.is thy_goal_stmtN
  orf Pred.is thy_goal_defnN)

(* APPLY-STYLE PROOFS *)

(* 
val prf_script_goalN = "prf_script_goal" (* i.e. subgoal *)
val prf_declN = "prf_decl" (* e.g. unfolding *)
val qed_scriptN = "qed_script" (* i.e. done *)

val qed_globalN = "qed_global" (* i.e. oops *) 
*)
val prf_scriptN = "prf_script" (* e.g. apply *)
val qedN = "qed" (* i.e. by *)
val unfoldingN = "unfolding"
val usingN = "using"
val includingN = "including"

fun starts_proof grpd_cmmnds = (Pred.is_not lemmasN)
  andf (is_in_group_where grpd_cmmnds (Pred.contains goalN))

fun ends_proof grpd_cmmnds = is_in_group_where grpd_cmmnds (Pred.contains qedN)

fun is_apply_step grpd_cmmnds = 
  (Pred.is_not qedN)
  andf (
    is_in_group_where grpd_cmmnds (Pred.disjunct [
      Pred.contains prf_scriptN, 
      Pred.contains qedN])
    orf (Pred.is unfoldingN)
    orf (Pred.is usingN)
    orf (Pred.is includingN)
  );

(* ISAR-STYLE PROOFS *)
(* 
val prf_asm_goalN = "prf_asm_goal" (* i.e. show, obtain and thus *)
val prf_blockN = "prf_block" (* i.e. proof *)
val prf_asmN = "prf_asm" (* e.g. define *)
val prf_closeN = "prf_close" (* i.e. { *)
val prf_openN = "prf_open" (* i.e. } *)
val prf_chainN = "prf_chain" (* e.g. ultimately *)
val qed_blockN = "qed_block" (* i.e. qed *) 
*)
val prf_goalN = "prf_goal" (* e.g. have *)

fun declares_local_goal grpd_cmmnds = is_in_group_where grpd_cmmnds (Pred.is prf_goalN)

val next_blockN = "next_block" (* i.e. next *)
val prfN = "prf"

fun is_isar_step grpd_cmmnds = 
  is_in_group_where grpd_cmmnds (Pred.contains prfN orf Pred.is next_blockN)
  andf (Pred.neg (is_apply_step grpd_cmmnds));

(* DOCUMENT PREPARATION *)
(* 
val document_headingN = "document_heading" (* e.g. section *)
val document_bodyN = "document_body" (* e.g. text *)
val document_rawN = "document_raw" (* i.e. text_raw *) 
*)
val documentN = "document"

fun is_document_step grpd_cmmnds = 
  is_in_group_where grpd_cmmnds (Pred.contains documentN);


end;