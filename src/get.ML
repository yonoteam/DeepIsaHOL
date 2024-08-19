
signature GET =
sig
  (* THEORY INFORMATION *)
  val transitions: theory -> Position.T -> string -> Toplevel.transition list
  val minor_keywords: theory -> string list
  val major_keywords: theory -> string list
  val command_keywords: theory -> string list
  val grouped_commands: theory -> (string * string list) list
  val deps: theory -> thm list -> (Thm_Name.T * thm) list
  val tokens: Proof.context -> string -> Token.T list
  val methods: Proof.context -> string list

  (* THEOREM INFORMATION *)
  val new_thms: Toplevel.state -> Toplevel.state -> (string * thm list) list
  val named_thms: (string * thm list) list -> (Thm_Name.T * thm) list
  val name_of_thm: thm -> Thm_Name.T
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
end;


structure Get: GET =
struct


(** THEORY INFORMATION **)

(* transitions: theory -> Position.T -> string -> Toplevel.transition list *)
fun transitions thy0 pos0 thy_text = Outer_Syntax.parse_text thy0 (K thy0) pos0 thy_text;

(* minor_keywords: theory -> string list *)
fun minor_keywords thy = Scan.dest_lexicon (Keyword.minor_keywords (Thy_Header.get_keywords thy));

(* major_keywords: theory -> string list *)
fun major_keywords thy = Scan.dest_lexicon (Keyword.major_keywords (Thy_Header.get_keywords thy));

(* command_keywords: theory -> string list *)
fun command_keywords thy = Keyword.dest_commands (Thy_Header.get_keywords thy);

(* grouped_commands: theory -> (string * string list) list *)
fun grouped_commands thy =
  let
    val kwrds = Thy_Header.get_keywords thy;
    fun get_kind name = the (Keyword.command_kind kwrds name);
    fun add_command cmnd [] = [(get_kind cmnd, [cmnd])]
      | add_command cmnd ((kind, cmnds) :: kcmndss) =
            if kind = get_kind cmnd then (kind, cmnd :: cmnds) :: kcmndss
            else (kind, cmnds) :: add_command cmnd kcmndss;
  in fold add_command (Keyword.dest_commands kwrds) [] end;

(* tokens: context -> string -> Token.T list *)
(* TODO: check if at right level *)
fun tokens ctxt = Token.explode (Thy_Header.get_keywords' ctxt) Position.none;

(* methods: Proof.context -> string list *)
fun methods ctxt = Name_Space.get_names (Method.method_space (Context.Proof ctxt));


(** THEOREM INFORMATION **)

(* new_thms: Toplevel.state -> Toplevel.state -> (string * thm list) list *)
fun new_thms state1 state2 =
  if not (Toplevel.is_toplevel state1) andalso not (Toplevel.is_toplevel state2) then
  let
    val generic1 = Toplevel.generic_theory_of state1;
    val generic2 = Toplevel.generic_theory_of state2;
    val facts1 = Context.cases Global_Theory.facts_of Proof_Context.facts_of generic1;
    val facts2 = Context.cases Global_Theory.facts_of Proof_Context.facts_of generic2;
    fun is_new (name, _) = (case Facts.lookup generic1 facts1 name of
        SOME _ => false 
      | NONE => true)
  in Facts.fold_static (Ops.insert_if is_new) facts2 [] end
  else raise Fail "new_thms: at top level";

(* named_thms: (string * thm list) list -> (Thm_Name.T * thm) list
 assumes input names are correct for the input thms (~ Thm_Name.make_list) *)
fun named_thms nthms =
  let
    fun distrib _ _ [] = []
      | distrib name n (thm :: thms) = ((name, n), thm) :: distrib name (n+1) thms;
  in maps (fn (name, thms) => distrib name 1 thms) nthms end;

(* name_of_thm: thm -> Thm_Name.T (c.f. Thm.derivation_name) *)
val name_of_thm = Thm.derivation_name

(* get_all_facts: Proof.context -> {global_f: Facts.T, local_f: Facts.T} *)
fun get_all_facts ctxt =
  let
    val thy = Proof_Context.theory_of ctxt;
    val local_facts = Proof_Context.facts_of ctxt;
    val global_facts = Global_Theory.facts_of thy;
  in {global_f = global_facts, local_f=local_facts} end;

(* passes_fact_check: Proof.context -> string -> bool *)
fun passes_fact_check ctxt thm_name =
  let
    val generic = Context.Proof ctxt
    val local_fs = #local_f (get_all_facts ctxt);
    val checks 
      = (case (Facts.check generic local_fs (thm_name, Position.none)) 
        of _ => true)
      handle ERROR _ => false
  in checks end;

(* get_all_thms: Proof.context -> (string * thm list) list *)
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

(* filtered_thms: (string * thm list -> bool) list -> Proof.context -> (string * thm list) list *)
fun filtered_thms filters ctxt =
  let
    val all_thms = get_all_thms ctxt;
    val filtered_thms = filter (Pred.conjunct filters) all_thms;
  in filtered_thms end;

(* thms: Proof.context -> string -> thm list *)
fun thms ctxt name =
  let
    val toks = tokens ctxt (name ^ " ")
    val refs_toks = (fst o Parse.thms1) toks;
    val thms = Attrib.eval_thms ctxt refs_toks
  in thms end;

(* deps: theory -> thm list -> (Thm_Name.T * thm) list *)
fun deps thy thms =
  let
    val lookup = Global_Theory.lookup_thm_id thy;
    fun thm_of_name thm_name = Global_Theory.get_thm_name thy (thm_name, Position.none);
    fun make_name_thm id_name = (snd id_name, thm_of_name (snd id_name));
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


(** THM INFORMATION **)

local

fun not_member_cons_if P x xs = Ops.insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

(* is_gen_var: term -> bool *)
fun is_gen_var (Term.Free _) = true
  | is_gen_var (Term.Var _) = true
  | is_gen_var _ = false;

(* is_gen_type_var: typ -> bool *)
fun is_gen_type_var (Term.TFree _) = true
  | is_gen_type_var (Term.TVar _) = true
  | is_gen_type_var _ = false;

(* add_atomic_types: typ -> typ list *) 
fun add_atomic_types T Ts = Term.fold_atyps (not_member_cons_if is_gen_type_var) T Ts;

(* get_uniques: (term -> 'a list) -> term list -> 'a list *)
fun get_uniques f (terms: term list) =
  let
    fun cond_add t ts = fold (not_member_cons_if (fn _ => true)) ts (f t);
  in fold cond_add terms [] end;

in

(* vars_of: term -> term list *)
fun vars_of t = Term.fold_aterms (not_member_cons_if is_gen_var) t [];

(* consts_of: term -> term list *)
fun consts_of t = Term.fold_aterms (not_member_cons_if Term.is_Const) t [];

(* atomic_types_of: term -> typ list *)
fun atomic_types_of t = Term.fold_types add_atomic_types t [];

(* vars: term list -> term list *)
val vars = get_uniques (vars_of)

(* consts: term list -> term list *)
val consts = get_uniques (consts_of)

(* atomic_types: term list -> typ list *)
val atomic_types = get_uniques (atomic_types_of);

end;

end;
