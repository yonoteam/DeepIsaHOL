
signature GET =
sig
  (* THEORY FILE INFORMATION *)
  val transitions: theory -> Position.T -> string -> Toplevel.transition list
  val transitions_info: Toplevel.state -> theory -> string 
    -> {kind: string, pos: Position.T, state: Toplevel.state, text: string, warn: (exn * string) option} list

  (* THEORY INFORMATION *)
  val minor_keywords: theory -> string list
  val major_keywords: theory -> string list
  val command_keywords: theory -> string list
  val grouped_commands: theory -> (string * string list) list
  val start_proof_keywords: theory -> string list
  val end_proof_keywords: theory -> string list
  val deps: theory -> thm list -> (string * thm) list

  (* CONTEXT INFORMATION *)
  val methods: Proof.context -> string list
  val passes_fact_check: Proof.context -> string -> bool
  val filtered_thms: (string * thm list -> bool) list 
    -> Proof.context -> (string * thm list) list
  val thms: Proof.context -> string -> thm list

  (* TERM INFORMATION *)
  val vars_of: term -> term list
  val consts_of: term -> term list
  val atomic_types_of: term -> typ list
  val vars: term list -> term list
  val consts: term list -> term list
  val atomic_types: term list -> typ list
  val goal_data: Toplevel.state 
    -> {consts: term list, goal: term, hyps: term list, types: typ list, vars: term list}
end;


structure Get: GET =
struct


(* THEORY FILE INFORMATION *)

(* theory -> Position.T -> string -> Toplevel.transition list *)
fun transitions thy0 pos0 thy_text = Outer_Syntax.parse_text thy0 (K thy0) pos0 thy_text;

(* Toplevel.state -> string list -> Toplevel.transition 
  -> {kind: string, pos: Position.T, state: Toplevel.state, text: string, warn: (exn * string) option} *)
fun make_transition_info state syms tr =
  let
    val (s, err) = Toplevel.transition true tr state;
    val k = Toplevel.name_of tr;
    val p = Toplevel.pos_of tr;
    val t = implode syms;
  in {kind = k, pos = p, state = s, text  = t, warn  = err} end;

(* Toplevel.state -> theory -> string 
  -> {kind: string, pos: Position.T, state: Toplevel.state, text: string, warn: (exn * string) option} list *)
fun transitions_info st0 thy0 thy_text =
  let
    val transitions = transitions thy0 Position.start thy_text;
    fun dist tr1 tr2 = Option.valOf (Position.distance_of (Toplevel.pos_of tr1, Toplevel.pos_of tr2));
    fun record_info st syms [tr] = [make_transition_info st syms tr]
      | record_info _ _ [] = []
      | record_info st syms (tr1 :: tr2 :: trs) = 
          let 
            val (syms1, syms2) = Library.chop (dist tr1 tr2) syms;
            val info = make_transition_info st syms1 tr1;
          in info :: record_info (#state info) syms2 (tr2 :: trs) end
  in record_info st0 (Symbol.explode thy_text) transitions end;


(* THEORY INFORMATION *)

(* theory -> string list *)
fun minor_keywords thy = Scan.dest_lexicon (Keyword.minor_keywords (Thy_Header.get_keywords thy));

(* theory -> string list *)
fun major_keywords thy = Scan.dest_lexicon (Keyword.major_keywords (Thy_Header.get_keywords thy));

(* theory -> string list *)
fun command_keywords thy = Keyword.dest_commands (Thy_Header.get_keywords thy);

(* theory -> (string * string list) list *)
fun grouped_commands thy =
  let
    val kwrds = Thy_Header.get_keywords thy;
    fun get_kind name = the (Keyword.command_kind kwrds name);
    fun add_command cmnd [] = [(get_kind cmnd, [cmnd])]
      | add_command cmnd ((kind, cmnds) :: kcmndss) =
            if kind = get_kind cmnd then (kind, cmnd :: cmnds) :: kcmndss
            else (kind, cmnds) :: add_command cmnd kcmndss;
  in fold add_command (Keyword.dest_commands kwrds) [] end;

(* theory -> string list *)
fun start_proof_keywords thy =
  grouped_commands thy 
  |> filter (Pred.on_fst (Pred.contains "goal"))
  |> maps snd;

(* theory -> string list *)
fun end_proof_keywords thy =
  grouped_commands thy 
  |> filter (Pred.on_fst (Pred.contains "qed"))
  |> maps snd;


(* CONTEXT INFORMATION *)

(* context -> string -> Token.T list *)
fun make_default_tokens ctxt = Token.explode (Thy_Header.get_keywords' ctxt) Position.none;

(* Proof.context -> string list *)
fun methods ctxt = Name_Space.get_names (Method.method_space (Context.Proof ctxt));

(* Proof.context -> {global_f: Facts.T, local_f: Facts.T} *)
fun get_all_facts ctxt =
  let
    val thy = Proof_Context.theory_of ctxt;
    val local_facts = Proof_Context.facts_of ctxt;
    val global_facts = Global_Theory.facts_of thy;
  in {global_f = global_facts, local_f=local_facts} end;

(* Proof.context -> string -> bool *)
fun passes_fact_check ctxt thm_name =
  let
    val generic = Context.Proof ctxt
    val local_fs = #local_f (get_all_facts ctxt);
    val checks 
      = (case (Facts.check generic local_fs (thm_name, Position.none)) 
        of _ => true)
      handle ERROR _ => false
  in checks end;

(* Proof.context -> (string * thm list) list *)
fun get_all_thms ctxt =
  let
    val generic = Context.Proof ctxt;
    fun consolidate_then_filter_wrt prevs facts
      = Facts.dest_all generic false prevs facts;

    val fs = get_all_facts ctxt;
    val all_facts
      = (consolidate_then_filter_wrt [#global_f fs] (#local_f fs)) 
        @ (consolidate_then_filter_wrt [] (#global_f fs));
  in all_facts end;

(* (string * thm list -> bool) list -> Proof.context -> (string * thm list) list *)
fun filtered_thms filters ctxt =
  let
    val all_thms = get_all_thms ctxt;
    val filtered_thms = filter (Pred.conjunct filters) all_thms;
  in filtered_thms end;

(* Proof.context -> string -> thm list *)
fun thms ctxt name =
  let
    val toks = make_default_tokens ctxt (name ^ " ")
    val refs_toks = (fst o Parse.thms1) toks;
    val thms = Attrib.eval_thms ctxt refs_toks
  in thms end;

(*  theory -> thm list -> (string * thm) list *)
fun deps thy thms =
  let
    val lookup = Global_Theory.lookup_thm_id thy;
    fun thm_of_name thm_name = Global_Theory.get_thm_name thy (thm_name, \<^here>);
    fun make_name_thm id_name = (fst (snd id_name), thm_of_name (snd id_name));
    fun deps (i, thm_node) res =
      if Inttab.defined res i then res
      else
        let val thm_id = Proofterm.thm_id (i, thm_node) in
          (case lookup thm_id of
            SOME thm_name =>
              Inttab.update (i, SOME (thm_id, thm_name)) res
          | NONE =>
              Inttab.update (i, NONE) res
              |> fold deps (Proofterm.thm_node_thms thm_node))
        end;
  in
    (Inttab.build (fold (fold deps o Thm.thm_deps o Thm.transfer thy) thms), [])
      |-> Inttab.fold_rev (fn (_, SOME entry) => cons (make_name_thm entry) | _ => I)
  end;


(* THM INFORMATION *)

local

fun not_member_cons_if P x xs = Data_Ops.insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

(* term -> bool *)
fun is_gen_var (Term.Free _) = true
  | is_gen_var (Term.Var _) = true
  | is_gen_var _ = false;

(* typ -> bool *)
fun is_gen_type_var (Term.TFree _) = true
  | is_gen_type_var (Term.TVar _) = true
  | is_gen_type_var _ = false;

(* typ -> typ list *) 
fun add_atomic_types T Ts = Term.fold_atyps (not_member_cons_if is_gen_type_var) T Ts;

(* (term -> 'a list) -> term list -> 'a list *)
fun get_uniques f (terms: term list) =
  let
    fun cond_add t ts = fold (not_member_cons_if (fn _ => true)) ts (f t);
  in fold cond_add terms [] end;

in

(* term -> term list *)
fun vars_of t = Term.fold_aterms (not_member_cons_if is_gen_var) t [];

(* term -> term list *)
fun consts_of t = Term.fold_aterms (not_member_cons_if Term.is_Const) t [];

(* term -> typ list *)
fun atomic_types_of t = Term.fold_types add_atomic_types t [];

(* term list -> term list *)
val vars = get_uniques (vars_of)

(* term list -> term list *)
val consts = get_uniques (consts_of)

(* term list -> typ list *)
val atomic_types = get_uniques (atomic_types_of);

(* Toplevel.state 
  -> {consts: term list, goal: term, hyps: term list, tvars: typ list, vars: term list} *)
fun goal_data state = 
  let
    val prop = Proof.goal (Toplevel.proof_of state);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val xs = vars goal_hyps;
    val cs = consts goal_hyps;
    val Ts = atomic_types goal_hyps;
  in {goal = hd goal_hyps, consts = cs, vars = xs, types = Ts, hyps = tl goal_hyps} end;

end;

end;