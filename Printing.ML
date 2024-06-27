
signature PRINTING =
sig
  val enumerate: 'a list -> (int * 'a) list (* TODO: move *)
  val mk_default_tokens: Proof.context -> string -> Token.T list (* TODO: move *)
  val neg: ('a -> bool) -> 'a -> bool
  val split_by: string -> string -> string list
  val get_facts: Proof.context -> {global_f: Facts.T, local_f: Facts.T}
  val get_method_names: Proof.context -> string list
  val get_global_fact_names: Proof.context -> string list
  val list_thms: (string * thm list -> bool) list -> Proof.context -> (string * thm list) list
  val get_thms: Proof.context -> string -> thm list
end;


structure Printing: PRINTING =
struct

fun enumerate xs =
  let
    fun iter_enum _ [] = []
      | iter_enum n (y::ys) = (n, y) :: (iter_enum (n+1) ys)
  in iter_enum 0 xs end;

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

(* ('a -> bool) -> 'a -> bool *)
fun neg P x = not (P x)

(* ('a -> bool) list -> 'a -> bool *)
fun conjunct_preds Ps = foldl1 (op andf) Ps;

(* (string * thm list -> bool) list -> Proof.context -> (string * thm list) list *)
fun list_thms filters ctxt =
  let
    val all_thms = list_all_thms ctxt;
    val filtered_thms = filter (conjunct_preds filters) all_thms;
  in filtered_thms end;

(* Proof.context -> string -> thm list *)
fun get_thms ctxt name =
  let
    val toks = mk_default_tokens ctxt (name ^ " ")
    val refs_toks = (fst o Parse.thms1) toks;
    val thms = Attrib.eval_thms ctxt refs_toks
  in thms end;

end;