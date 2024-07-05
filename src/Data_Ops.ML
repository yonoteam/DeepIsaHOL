
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
  val vars_of: term -> term list
  val consts_of: term -> term list
  val atomic_types_of: term -> typ list
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

local

(* 'a * 'b -> ('a * 'b list) list -> ('a * 'b list) list *)
fun ins_entry (x, y) =
  AList.default (op =) (x, []) #>
  AList.map_entry (op =) x (insert (op =) y);

(* term -> (typ * (string * typ) list) list -> (typ * (string * typ) list) list *)
val add_consts = Term.fold_aterms
  (fn Const (c, T) => ins_entry (T, (c, T))
    | _ => I);

(* term -> (typ * (string * int) list) list -> (typ * (string * int) list) list *)
val add_vars = Term.fold_aterms
  (fn Free (x, T) => ins_entry (T, (x, ~1))
    | Var (xi, T) => ins_entry (T, xi)
    | _ => I);

(* typ -> (sort * (string * int) list) list -> (sort * (string * int) list) list *)
val add_varsT = Term.fold_atyps
  (fn TFree (x, S) => ins_entry (S, (x, ~1))
    | TVar (xi, S) => ins_entry (S, xi)
    | _ => I);

fun sort_idxs vs = map (apsnd (sort (prod_ord string_ord int_ord))) vs;
fun sort_cnsts cs = map (apsnd (sort_by fst)) cs;

fun not_member_of xs x = Pred.neg (member (op =) xs) x;
fun not_member_cons_if P x xs = insert_if (P andf not_member_of xs) x xs;

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

(* (term -> ''a list) -> term list -> ''a list *)
fun get_uniques f (terms: term list) =
  let
    fun cond_add t ts = fold (not_member_cons_if (fn _ => true)) ts (f t);
  in fold cond_add terms [] end;

(* (term -> (typ * (string * int) list) list) -> term list -> (typ * (string * int) list) list *)
fun get_flats f (terms: term list) =
  let
    fun cond_add_vars (key, vals) kvss = fold (fn v => fn kvs => ins_entry (key,v) kvs) vals kvss;
    fun add_vars_of_not_in t kvss = fold cond_add_vars kvss (f t);
  in fold add_vars_of_not_in terms [] end;

(* term -> (typ * (string * int) list) list *)
fun pseudo_vars_of t = sort_idxs (add_vars t []);

(* term -> (typ * (string * typ) list) list *)
fun pseudo_consts_of t = sort_cnsts (add_consts t []);

(* term -> (sort * (string * int) list) list *)
fun pseudo_types_of t = rev (sort_idxs (Term.fold_types add_varsT t []));

in

(* term -> term list *)
fun vars_of t = Term.fold_aterms (not_member_cons_if is_gen_var) t [];

(* term -> term list *)
fun consts_of t = Term.fold_aterms (not_member_cons_if Term.is_Const) t [];

(* term -> typ list *)
fun atomic_types_of t = Term.fold_types add_atomic_types t [];

(* term list -> term list *)
val get_vars = get_uniques (vars_of)

(* term list -> (typ * (string * int) list) list *)
val get_flat_vars = get_flats (pseudo_vars_of);

(* term list -> term list *)
val get_consts = get_uniques (consts_of)

(* term list -> (typ * (string * typ) list) list *)
val get_flat_consts = get_flats (pseudo_consts_of);

(* term list -> typ list *)
val get_atomic_types = get_uniques (atomic_types_of);

(* term list -> (sort * (string * int) list) list *)
val get_flat_types = get_flats (pseudo_types_of);

end;

end;