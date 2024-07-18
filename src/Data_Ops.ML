
signature DATA_OPS =
sig
  val insert_if: ('a -> bool) -> 'a -> 'a list -> 'a list
  val intersp: 'a list -> 'a list -> 'a list
  val enumerate: 'a list -> (int * 'a) list
  val mk_default_tokens: Proof.context -> string -> Token.T list
  val split_by: string -> string -> string list
  val uniques: ''a list -> ''a list
  val read_file: string -> string
  val get_facts: Proof.context -> {global_f: Facts.T, local_f: Facts.T}
  val passes_fact_check: Proof.context -> string -> bool
  val get_methods: Proof.context -> string list
  val get_keywords_minor: theory -> string list
  val get_keywords_major: theory -> string list
  val get_keywords_commands: theory -> string list
  val get_keywords_commands_grouped: theory -> (string * string list) list
  val get_global_fact_names: Proof.context -> string list
  val get_filtered_thms: (string * thm list -> bool) list 
    -> Proof.context -> (string * thm list) list
  val get_thms: Proof.context -> string -> thm list
  val get_deps: theory -> thm list -> (string * thm) list
  val get_transitions_info: Toplevel.state -> theory -> string 
  -> {kind: string, pos: Position.T, state: Toplevel.state, text: string, warn: (exn * string) option} list
  val vars_of: term -> term list
  val consts_of: term -> term list
  val atomic_types_of: term -> typ list
  val get_vars: term list -> term list
  val get_consts: term list -> term list
  val get_atomic_types: term list -> typ list
  val get_goal_data: Toplevel.state 
    -> {consts: term list, goal: term, hyps: term list, types: typ list, vars: term list}
end;


structure Data_Ops: DATA_OPS =
struct

(* GENERIC DATA OPS *)

(* ('a -> bool) -> 'a -> 'a list -> 'a list *)
fun insert_if P x xs = if P x then x :: xs else xs;

(* 'a list -> 'a list -> 'a list *)
fun intersp seps (x :: (xs as  _ :: _)) = (x :: seps) @ (intersp seps xs)
  | intersp _ xs = xs;

(* 'a list -> (int * 'a) list *)
fun enumerate xs =
  let
    fun iter_enum _ [] = []
      | iter_enum n (y::ys) = (n, y) :: (iter_enum (n+1) ys)
  in iter_enum 0 xs end;

(* string -> string -> string list *)
fun split_by split_str target = space_explode split_str target;

(* 'a list -> 'a list *)
fun uniques xs = Library.distinct (op =) xs;

(* string -> string *)
fun read_file fname = 
  let 
    val IO_instream = TextIO.openIn fname;
    val content = TextIO.inputAll IO_instream;
    val _ = TextIO.closeIn IO_instream;
  in content end;


(* CONTEXT RETRIEVAL INFORMATION *)

(* context -> string -> Token.T list *)
fun mk_default_tokens ctxt = Token.explode (Thy_Header.get_keywords' ctxt) Position.none;

(* Proof.context -> string list *)
fun get_methods ctxt = Name_Space.get_names (Method.method_space (Context.Proof ctxt));

(* theory -> string list *)
fun get_keywords_minor thy = Scan.dest_lexicon (Keyword.minor_keywords (Thy_Header.get_keywords thy));

(* theory -> string list *)
fun get_keywords_major thy = Scan.dest_lexicon (Keyword.major_keywords (Thy_Header.get_keywords thy));

(* theory -> string list *)
fun get_keywords_commands thy = Keyword.dest_commands (Thy_Header.get_keywords thy);

(* theory -> (string * string list) list *)
fun get_keywords_commands_grouped thy =
  let
    val kwrds = Thy_Header.get_keywords thy;
    fun get_kind name = the (Keyword.command_kind kwrds name);
    fun add_command cmnd [] = [(get_kind cmnd, [cmnd])]
      | add_command cmnd ((kind, cmnds) :: kcmndss) =
            if kind = get_kind cmnd then (kind, cmnd :: cmnds) :: kcmndss
            else (kind, cmnds) :: add_command cmnd kcmndss;
  in fold add_command (Keyword.dest_commands kwrds) [] end;

(* context -> string list *)
fun get_global_fact_names ctxt = Name_Space.get_names 
  (Facts.space_of (Global_Theory.facts_of (Proof_Context.theory_of ctxt)));

(* Proof.context -> {global_f: Facts.T, local_f: Facts.T} *)
fun get_facts ctxt =
  let
    val thy = Proof_Context.theory_of ctxt;
    val local_facts = Proof_Context.facts_of ctxt;
    val global_facts = Global_Theory.facts_of thy;
  in {global_f = global_facts, local_f=local_facts} end;

(* Proof.context -> string -> bool *)
fun passes_fact_check ctxt thm_name =
  let
    val generic = Context.Proof ctxt
    val facts = #local_f (get_facts ctxt);
    val checks 
      = (case (Facts.check generic facts (thm_name, Position.none)) 
        of _ => true)
      handle ERROR _ => false
  in checks end;

(* Proof.context -> (string * thm list) list *)
fun get_all_thms ctxt =
  let
    val generic = Context.Proof ctxt;
    fun consolidate_then_filter_wrt prevs facts
      = Facts.dest_all generic false prevs facts;

    val facts = get_facts ctxt;
    val all_facts
      = (consolidate_then_filter_wrt [#global_f facts] (#local_f facts)) 
        @ (consolidate_then_filter_wrt [] (#global_f facts));
  in all_facts end;

(* (string * thm list -> bool) list -> Proof.context -> (string * thm list) list *)
fun get_filtered_thms filters ctxt =
  let
    val all_thms = get_all_thms ctxt;
    val filtered_thms = filter (Pred.conjunct filters) all_thms;
  in filtered_thms end;

(* Proof.context -> string -> thm list *)
fun get_thms ctxt name =
  let
    val toks = mk_default_tokens ctxt (name ^ " ")
    val refs_toks = (fst o Parse.thms1) toks;
    val thms = Attrib.eval_thms ctxt refs_toks
  in thms end;

(*  theory -> thm list -> (string * thm) list *)
fun get_deps thy =
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
    fn thms =>
      (Inttab.build (fold (fold deps o Thm.thm_deps o Thm.transfer thy) thms), [])
      |-> Inttab.fold_rev (fn (_, SOME entry) => cons (make_name_thm entry) | _ => I)
  end;


(* THEORY FILE INFORMATION *)

(* theory -> Position.T -> string -> Toplevel.transition list *)
fun make_transitions thy pos0 str = Outer_Syntax.parse_text thy (K thy) pos0 str;

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
fun get_transitions_info st0 thy0 thy_text =
  let
    val transitions = make_transitions thy0 Position.start thy_text;
    fun dist tr1 tr2 = Option.valOf (Position.distance_of (Toplevel.pos_of tr1, Toplevel.pos_of tr2));
    fun record_info st syms [tr] = [make_transition_info st syms tr]
      | record_info _ _ [] = []
      | record_info st syms (tr1 :: tr2 :: trs) = 
          let 
            val (syms1, syms2) = Library.chop (dist tr1 tr2) syms;
            val info = make_transition_info st syms1 tr1;
          in info :: record_info (#state info) syms2 (tr2 :: trs) end
  in record_info st0 (Symbol.explode thy_text) transitions end;


(* THM INFORMATION *)

local

fun not_member_cons_if P x xs = insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

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
val get_vars = get_uniques (vars_of)

(* term list -> term list *)
val get_consts = get_uniques (consts_of)

(* term list -> typ list *)
val get_atomic_types = get_uniques (atomic_types_of);

(* Toplevel.state 
  -> {consts: term list, goal: term, hyps: term list, tvars: typ list, vars: term list} *)
fun get_goal_data state = 
  let
    val prop = Proof.goal (Toplevel.proof_of state);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val xs = get_vars goal_hyps;
    val cs = get_consts goal_hyps;
    val Ts = get_atomic_types goal_hyps;
  in {goal = hd goal_hyps, consts = cs, vars = xs, types = Ts, hyps = tl goal_hyps} end;

end;

end;