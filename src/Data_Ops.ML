
signature DATA_OPS =
sig
  val insert_if: ('a -> bool) -> 'a -> 'a list -> 'a list
  val enumerate: 'a list -> (int * 'a) list
  val mk_default_tokens: Proof.context -> string -> Token.T list
  val split_by: string -> string -> string list
  val get_facts: Proof.context -> {global_f: Facts.T, local_f: Facts.T}
  val passes_fact_check: Proof.context -> string -> bool
  val get_methods: Proof.context -> string list
  val get_global_fact_names: Proof.context -> string list
  val get_filtered_thms: (string * thm list -> bool) list -> Proof.context -> (string * thm list) list
  val get_thms: Proof.context -> string -> thm list
  val get_deps: theory -> thm list -> (string * thm) list
  val get_vars: term list -> term list
  val get_consts: term list -> term list
  val get_atomic_types: term list -> typ list
end;


structure Data_Ops: DATA_OPS =
struct

(* GENERIC DATA OPS *)

(* ('a -> bool) -> 'a -> 'a list -> 'a list *)
fun insert_if P x xs = if P x then x :: xs else xs;

(* 'a list -> (int * 'a) list *)
fun enumerate xs =
  let
    fun iter_enum _ [] = []
      | iter_enum n (y::ys) = (n, y) :: (iter_enum (n+1) ys)
  in iter_enum 0 xs end;

(* string -> string -> string list *)
fun split_by split_str target = space_explode split_str target;


(* CONTEXT RETRIEVAL INFORMATION *)

(* context -> string -> Token.T list *)
fun mk_default_tokens ctxt = Token.explode (Thy_Header.get_keywords' ctxt) Position.none;

fun get_methods ctxt = Name_Space.get_names (Method.method_space (Context.Proof ctxt));

(* context -> string list *)
fun get_global_fact_names ctxt = Name_Space.get_names (Facts.space_of (Global_Theory.facts_of (Proof_Context.theory_of ctxt)));

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
in
  checks
end;

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


(* THM INFORMATION *)

(* term -> bool *)
fun is_gen_var (Term.Free _) = true
  | is_gen_var (Term.Var _) = true
  | is_gen_var _ = false;

fun not_member_of ts t = Pred.neg (member (op =) ts) t;

(* term list -> term list *)
fun get_vars terms = 
  let
    fun cond_add t ts = insert_if (is_gen_var andf not_member_of ts) t ts;
    fun cond_add_vars t vs = fold (fn var => cond_add var) vs (Term.fold_aterms cond_add t []);
  in fold cond_add_vars terms [] end;

(* term -> term list *)
fun get_consts terms = 
  let
    fun cond_add t ts = insert_if (is_Const andf not_member_of ts) t ts;
    fun cond_add_consts t cs = fold (fn c => cond_add c) cs (Term.fold_aterms cond_add t []);
  in fold cond_add_consts terms [] end;

(* typ -> bool *)
fun is_gen_type_var (Term.TFree _) = true
  | is_gen_type_var (Term.TVar _) = true
  | is_gen_type_var _ = false;

(* term -> typ list *)
fun get_atomic_types terms =
  let
    fun cond_add T Ts = insert_if (is_gen_type_var andf not_member_of Ts) T Ts;
    fun add_subtypes T Ts = Term.fold_atyps cond_add T Ts;
    fun cond_add_typs t Ts = fold (fn T => cond_add T) Ts (Term.fold_types add_subtypes t []);
  in fold cond_add_typs terms []  end;

end;