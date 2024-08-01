
signature DATA =
sig
  (* data *)
  type data
  val build_data: {
    action: string, 
    consts: term list, 
    facts: Proof.context * (Thm_Name.T * thm) list, 
    goal: term, 
    hyps: term list, 
    methods: string list, 
    state: Toplevel.state, 
    types: typ list, 
    vars: term list} 
    -> data
  val action_of: data -> string
  val state_of: data -> Toplevel.state
  val goal_of: data -> term
  val hyps_of: data -> term list
  val consts_of: data -> term list
  val vars_of: data -> term list
  val types_of: data -> typ list
  val facts_of: data -> Proof.context * (Thm_Name.T * thm) list
  val methods_of: data -> string list
 
  (* Extraction *)
  val find_deps: (Toplevel.state * Actions.T) Sections.T -> Proof.context * (Thm_Name.T * thm) list
  val extract_data: (Toplevel.state * Actions.T) Sections.T -> data list
  val make_steps: (Toplevel.state * Actions.T) Sections.T -> data list list
  
  (* Pretty printing *)
  val pretty_json_action: data -> Pretty.T
  val pretty_json_goal: Proof.context -> data -> Pretty.T
  val pretty_json_consts: bool -> int -> Proof.context -> data -> Pretty.T
  val pretty_json_vars: bool -> int -> Proof.context -> data -> Pretty.T
  val pretty_json_types: bool -> int -> Proof.context -> data -> Pretty.T
  val pretty_json_facts: bool -> int -> data -> Pretty.T
  val pretty_json_data: bool -> data -> Pretty.T
end;


structure Data: DATA =
struct

datatype data = Data of {
  action: string, 
  consts: term list, 
  facts: Proof.context * (Thm_Name.T * thm) list, 
  goal: term, 
  hyps: term list, 
  methods: string list, 
  state: Toplevel.state, 
  types: typ list, 
  vars: term list
  };

fun build_data d = Data d;
fun rep_data (Data d) = d;
fun action_of d = #action (rep_data d);
fun state_of d = #state (rep_data d);
fun goal_of d = #goal (rep_data d);
fun hyps_of d = #hyps (rep_data d);
fun consts_of d = #consts (rep_data d);
fun vars_of d = #vars (rep_data d);
fun types_of d = #types (rep_data d);
fun facts_of d = #facts (rep_data d);
fun methods_of d = #methods (rep_data d);

(* (Toplevel.state * Actions.T) Sections.T -> context * (Thm_Name.T * thm) list *)
fun find_deps proof =
  let 
    val (s0, _) = Sections.get_first proof;
    val (sN, aN) = Sections.get_last proof;
    val sN' = #1 (Actions.apply aN sN);
    val thyN' = Toplevel.theory_of sN'
    val diffs = Get.new_thms s0 sN'
  in (Toplevel.context_of sN', Get.deps thyN' (maps snd diffs)) end;

(* (Toplevel.state * Actions.T) Sections.T -> data list *)
fun extract_data proof =
  let
    val lemmas = find_deps proof;
    val raw_pairs = tl (Sections.recover_trace proof);
    fun make_data (s, a) =
      let
        val prop = Proof.goal (Toplevel.proof_of s);
        val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
        val xs = Get.vars goal_hyps;
        val cs = Get.consts goal_hyps;
        val Ts = Get.atomic_types goal_hyps;
        val mthds = Get.methods (Toplevel.context_of s);
      in build_data {state = s, 
          action = Actions.text_of a,
          goal = hd goal_hyps, 
          consts = cs, 
          vars = xs, 
          types = Ts, 
          hyps = tl goal_hyps,
          methods = mthds,
          facts = lemmas} 
      end;
  in map make_data raw_pairs end;

fun make_steps sections =
  let
    val proofs = Sections.get_topmost Sections.proof_tag sections;
  in map (fn proof => extract_data proof) proofs end;


(** PRETTY PRINTING **)


local

fun not_member_cons_if P x xs = Data_Ops.insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

(* ('a -> ('val, 'key)) -> 'a list -> ('key * 'val list) list *)
fun alist_by extract_info xs =
let
  fun register x [] = 
      let
        val (name, key) = extract_info x;
      in [(key, [name])] end
    | register x ((key', names) :: kns) =
      let
        val (name, key) = extract_info x;
      in if key' = key 
        then (key', not_member_cons_if (fn _ => true) name names) :: kns  
        else (key', names) :: register x kns end;
in fold register xs [] end;

(* term -> term * typ *)
fun const_info t = (case t of 
  Const (_, T) => (t, T) 
  | _ => raise TERM("not a constant", [t]));

(* term -> term * typ *)
fun vars_info t = (case t of 
  Var (_, T) => (t,T)  
  | Free (_, T) => (t, T) 
  | _ => raise TERM("not a var", [t]));

(* typ -> typ * sort *)
fun types_info T = (case T of 
  TVar (_, S) => (T, S) 
  | TFree (_, S) => (T, S) 
  | _ => raise TYPE("not a type variable", [T], []));

(* Proof.context -> (Proof.context -> 'key -> Pretty.T) -> (Proof.context -> 'val -> Pretty.T) 
  -> string -> ('key * 'val list) list -> Pretty.T list *)
fun pretty_alist ctxt pretty_key pretty_val str alist = 
  let
    fun pretty_right k vs = Pretty.block (
      Pretty.str "\"" :: Pretty.commas (map (pretty_val ctxt) vs) 
      @ [Pretty.brk 1, Pretty.str "::", Pretty.brk 1, (pretty_key ctxt) k, Pretty.str "\""]
    );
    fun pretty_entry (n, (k, vs)) = Print.pretty_json_elem (str^(Value.print_int n)) (pretty_right k vs)
  in map pretty_entry (Data_Ops.enumerate alist) end;

(* Proof.context -> term list -> Pretty.T *)
fun pretty_consts ctxt consts = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by const_info consts);

(* Proof.context -> term list -> Pretty.T *)
fun pretty_vars ctxt vars = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by vars_info vars);

(* Proof.context -> typ list -> Pretty.T list *)
fun pretty_types ctxt typs = pretty_alist ctxt
  Syntax.pretty_sort Syntax.pretty_typ "Sort" (alist_by types_info typs);

(* Proof.context -> (string * int) * thm -> Pretty.T list *)
fun pretty_fact ctxt ((name, i),thm) = [
  Print.pretty_json_elem "name" (
    Pretty.quote (Pretty.block [
      Pretty.marks_str (#1 (Proof_Context.markup_extern_fact ctxt name), name),
      Pretty.str (Thm_Name.print_suffix (name, i))])
  ),
  Print.pretty_json_elem "term" (Pretty.quote (Thm.pretty_thm ctxt thm))
  ];

(* Proof.context -> int -> ((string * int) * thm) list -> Pretty.T *)
fun pretty_facts debug lvl (ctxt, named_thms) =
  let
    val prts = map (fn fact => 
      Print.pretty_json_elem "thm"
        (Print.pretty_json_attribs false "{" "}" (~2) (pretty_fact ctxt fact))) named_thms
  in Print.pretty_json_elem "thms" (Print.pretty_json_attribs debug "[" "]" lvl prts) end;

(* ('a -> 'b) -> ('c -> 'b -> Pretty.T list) -> bool -> 'c -> string -> 'a -> Pretty.T *)
fun to_pretty retriever prettyfier debug lvl ctxt name data =
  let
    val jattribs = Print.pretty_json_attribs debug "[" "]" lvl (prettyfier ctxt (retriever data))
  in Print.pretty_json_elem name jattribs end;

in 

fun pretty_json_action data = Print.pretty_json_elem "action" (Pretty.quote (Pretty.str (action_of data)))
fun pretty_json_goal ctxt data = Print.pretty_json_elem "term" (Pretty.quote (Syntax.pretty_term ctxt (goal_of data)));
fun pretty_json_consts debug lvl ctxt = to_pretty consts_of pretty_consts debug lvl ctxt "constants"
fun pretty_json_vars debug lvl ctxt = to_pretty vars_of pretty_vars debug lvl ctxt "variables"
fun pretty_json_types debug lvl ctxt = to_pretty types_of pretty_types debug lvl ctxt "type variables"
fun pretty_json_facts debug lvl data = pretty_facts debug lvl (facts_of data)

(* bool -> Toplevel.state -> Proof.context * ((string * int) * thm) list -> Pretty.T *)
fun pretty_json_data debug data = 
  let
    val state = state_of data;
    val ctxt = Proof.context_of (Toplevel.proof_of state)
      |> Config.put show_sorts false
      |> Config.put show_types false 
    fun to_pretty retriever prettyfier name =
      let
        val jattribs = Print.pretty_json_attribs debug "[" "]" 4 (prettyfier ctxt (retriever data))
      in Print.pretty_json_elem name jattribs end;
    val Ts = to_pretty types_of pretty_types "type variables"
    val cs = to_pretty consts_of pretty_consts "constants"
    val vs = to_pretty vars_of pretty_vars "variables"
    val goal = Print.pretty_json_elem "term" (Pretty.quote (Syntax.pretty_term ctxt (goal_of data)));
    val lems = pretty_facts debug 4 (facts_of data);
    val pretty_state = Print.pretty_json_attribs debug "{" "}" 2 [goal, vs, cs, Ts, lems]
  in Print.pretty_json_attribs debug "{" "}" 0 [Print.pretty_json_elem "state" pretty_state] end;

(* mode -> bool -> Toplevel.state -> unit *)
(* fun proof_state mode debug state = write_to mode (string_of (pretty_state_no_facts debug state));
 *)

end

end;
