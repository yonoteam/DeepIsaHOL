
signature DATA_OPS =
sig
  val enumerate: 'a list -> (int * 'a) list (* TODO: move *)
  val mk_default_tokens: Proof.context -> string -> Token.T list (* TODO: move *)
  val split_by: string -> string -> string list
  val get_facts: Proof.context -> {global_f: Facts.T, local_f: Facts.T}
  val get_method_names: Proof.context -> string list
  val get_global_fact_names: Proof.context -> string list
  val list_thms: (string * thm list -> bool) list -> Proof.context -> (string * thm list) list
  val get_thms: Proof.context -> string -> thm list
  val get_deps: theory -> thm list -> (string * thm) list
end;


structure Data_Ops: DATA_OPS =
struct

fun enumerate xs =
  let
    fun iter_enum _ [] = []
      | iter_enum n (y::ys) = (n, y) :: (iter_enum (n+1) ys)
  in iter_enum 0 xs end;

(* context -> string -> Token.T list *)
fun mk_default_tokens ctxt = Token.explode (Thy_Header.get_keywords' ctxt) Position.none;

fun split_by splitter target = space_explode splitter target;

fun get_method_names ctxt = Name_Space.get_names (Method.method_space (Context.Proof ctxt));

(* context -> string list *)
fun get_global_fact_names ctxt = Name_Space.get_names (Facts.space_of (Global_Theory.facts_of (Proof_Context.theory_of ctxt)));

(* Proof.context -> {global_f: Facts.T, local_f: Facts.T} *)
fun get_facts ctxt =
  let
    val thy = Proof_Context.theory_of ctxt;
    val local_facts = Proof_Context.facts_of ctxt;
    val global_facts = Global_Theory.facts_of thy;
  in {global_f = global_facts, local_f=local_facts} end;

(* Proof.context -> (string * thm list) list *)
fun list_all_thms ctxt =
  let
    val generic = Context.Proof ctxt;
    fun consolidate_then_filter_wrt prevs facts
      = Facts.dest_all generic false prevs facts;

    val facts = get_facts ctxt;
    val all_facts
      = (consolidate_then_filter_wrt [#global_f facts] (#local_f facts)) 
        @ (consolidate_then_filter_wrt [] (#global_f facts));
  in all_facts end;

(* ('a -> bool) list -> 'a -> bool *)
fun conjunct Ps = foldl1 (op andf) Ps;

(* (string * thm list -> bool) list -> Proof.context -> (string * thm list) list *)
fun list_thms filters ctxt =
  let
    val all_thms = list_all_thms ctxt;
    val filtered_thms = filter (conjunct filters) all_thms;
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

end;