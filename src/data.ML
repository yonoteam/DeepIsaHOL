
signature DATA =
sig
  (* Data.T *)
  type T
  val init: Toplevel.state -> Actions.T -> T
  val build: Proof.context * (Thm_Name.T * thm) list -> Toplevel.state -> Actions.T -> T
  val state_of: T -> Toplevel.state
  val goal_of: T -> term
  val hyps_of: T -> term list
  val consts_of: T -> term list
  val vars_of: T -> term list
  val types_of: T -> typ list
  val facts_of: T -> Proof.context * (Thm_Name.T * thm) list
  val methods_of: T -> string list
  val from: Toplevel.state -> T
  val action_of: T -> string
  val apply: string -> Toplevel.state -> T

  (* Extraction *)
  val find_deps: (Toplevel.state * Actions.T) Sections.T -> Proof.context * (Thm_Name.T * thm) list
  val extract: (Toplevel.state * Actions.T) Sections.T -> T list
  val extract_proofs: (Toplevel.state * Actions.T) Sections.T -> T list list
  
  (* Pretty printing *)
  val pretty_json_action: T -> Pretty.T
  val pretty_json_goal: Proof.context -> T -> Pretty.T
  val pretty_json_hyps: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_consts: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_vars: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_types: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_facts: bool -> int -> T -> Pretty.T
  val pretty_json_methods: bool -> int -> T -> Pretty.T
  val pretty_json: bool -> T -> Pretty.T
  val pretty_apply: string -> Toplevel.state -> Pretty.T

  (* IO *)
  val print_out: Toplevel.state -> unit
  val write_proof: string -> int -> T list -> unit
  val write_proofs: string -> T list list -> unit
end;


structure Data: DATA =
struct


(** DATA **)

datatype T = Data of {
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

(* Data.T -> _ *)
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

(* init: Toplevel.state -> Actions.T -> Data.T *)
fun init s a =
  let
    val prop = Proof.goal (Toplevel.proof_of s);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val xs = Get.vars goal_hyps;
    val cs = Get.consts goal_hyps;
    val Ts = Get.atomic_types goal_hyps;
  in Data {state = s, 
      action = Actions.text_of a,
      goal = hd goal_hyps, 
      consts = cs, 
      vars = xs, 
      types = Ts, 
      hyps = tl goal_hyps,
      methods = [],
      facts = (Toplevel.context_of s, [])} 
  end;

(* build: Proof.context * (Thm_Name.T * thm) list -> Toplevel.state -> Actions.T -> Data.T *)
fun build fs s a = 
  let
    val prop = Proof.goal (Toplevel.proof_of s);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val xs = Get.vars goal_hyps;
    val cs = Get.consts goal_hyps;
    val Ts = Get.atomic_types goal_hyps;
    val mthds = 
      let
        val grp_cmmnds = Get.grouped_commands (Toplevel.theory_of s);
        fun is_apply_step act = Actions.on_kind (Seps.is_apply_step grp_cmmnds) act;
      in if is_apply_step a then Get.methods (Toplevel.context_of s) else [] end;
  in Data {state = s, 
      action = Actions.text_of a,
      goal = hd goal_hyps, 
      consts = cs, 
      vars = xs, 
      types = Ts, 
      hyps = tl goal_hyps,
      methods = mthds,
      facts = fs} 
  end;

(* from: Toplevel.state -> Data.T *)
fun from s = build (Toplevel.context_of s, []) s (Actions.void);

(* apply: string -> Toplevel.state -> Data.T *)
(* TODO: make it Data.T -> Data.T*)
fun apply txt s = 
  let
    val a = hd (Actions.make (Toplevel.theory_of s) txt);
    val s' = Actions.apply a s;
  in build (Toplevel.context_of s, []) s' (Actions.void) end;

(** EXTRACTION **)
(* Assumes the input section is well-formed and that it is a proof *) 
(* find_deps: (Toplevel.state * Actions.T) Sections.T -> context * (Thm_Name.T * thm) list *)
fun find_deps proof =
  let 
    val (s0, _) = Sections.get_first proof;
    val (sN, aN) = Sections.get_last proof;
    val sN' = Actions.apply aN sN;
    val thyN' = Toplevel.theory_of sN'
    val diffs = Get.new_thms s0 sN'
  in (Toplevel.context_of sN', Get.deps thyN' (maps snd diffs)) end;

(* Assumes the input section is well-formed and that it is a proof *) 
(* extract: (Toplevel.state * Actions.T) Sections.T -> data list *)
fun extract proof =
  let
    val lemmas = find_deps proof;
    val raw_pairs = tl (Sections.recover_trace proof);
    fun make_data (s, a) = build lemmas s a;
  in map make_data raw_pairs end;

(* extract_proofs: (Toplevel.state * Actions.T) Sections.T -> data list list *)
fun extract_proofs sections =
  let
    val proofs = Sections.get_topmost Sections.proof_tag sections;
  in map (fn proof => extract proof) proofs end;


(** PRETTY PRINTING **)

local

fun not_member_cons_if P x xs = Ops.insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

(* alist_by: ('a -> ('val, 'key)) -> 'a list -> ('key * 'val list) list *)
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

(* const_info: term -> term * typ *)
fun const_info t = (case t of 
  Const (_, T) => (t, T) 
  | _ => raise TERM("not a constant", [t]));

(* vars_info: term -> term * typ *)
fun vars_info t = (case t of 
  Var (_, T) => (t,T)  
  | Free (_, T) => (t, T) 
  | _ => raise TERM("not a var", [t]));

(* types_info: typ -> typ * sort *)
fun types_info T = (case T of 
  TVar (_, S) => (T, S) 
  | TFree (_, S) => (T, S) 
  | _ => raise TYPE("not a type variable", [T], []));

(* pretty_alist: Proof.context 
  -> (Proof.context -> 'key -> Pretty.T) 
  -> (Proof.context -> 'val -> Pretty.T) 
  -> string -> ('key * 'val list) list -> Pretty.T list *)
fun pretty_alist ctxt pretty_key pretty_val str alist = 
  let
    fun pretty_right k vs = Pretty.block (
      Pretty.str "\"" :: Pretty.commas (map (pretty_val ctxt) vs) 
      @ [Pretty.brk 1, Pretty.str "::", Pretty.brk 1, (pretty_key ctxt) k, Pretty.str "\""]
    );
    fun pretty_entry (n, (k, vs)) 
      = Print.pretty_json_elem (str^(Value.print_int n)) (pretty_right k vs)
  in map pretty_entry (Ops.enumerate alist) end;

(* pretty_consts: Proof.context -> term list -> Pretty.T *)
fun pretty_consts ctxt consts = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by const_info consts);

(* pretty_vars: Proof.context -> term list -> Pretty.T *)
fun pretty_vars ctxt vars = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by vars_info vars);

(* pretty_types: Proof.context -> typ list -> Pretty.T list *)
fun pretty_types ctxt typs = pretty_alist ctxt
  Syntax.pretty_sort Syntax.pretty_typ "Sort" (alist_by types_info typs);

(* pretty_methods: unit -> string list -> Pretty.T list *)
fun pretty_methods () = (map (Print.pretty_json_elem "name" o Pretty.quote o Pretty.str));

(* pretty_hyps: Proof.context -> string list -> Pretty.T list *)
fun pretty_hyps ctxt 
  = (map (Print.pretty_json_elem "name" o Pretty.quote o Syntax.pretty_term ctxt));

(* pretty_fact: Proof.context -> (string * int) * thm -> Pretty.T list *)
fun pretty_fact ctxt ((name, i),thm) = [
  Print.pretty_json_elem "name" (
    Pretty.quote (Pretty.block [
      Pretty.marks_str (#1 (Proof_Context.markup_extern_fact ctxt name), name),
      Pretty.str (Thm_Name.print_suffix (name, i))])
  ),
  Print.pretty_json_elem "term" (Pretty.quote (Thm.pretty_thm ctxt thm))
  ];

(* pretty_facts: Proof.context -> int -> ((string * int) * thm) list -> Pretty.T *)
fun pretty_facts debug lvl (ctxt, named_thms) =
  let
    val prts = map (fn fact => 
      Print.pretty_json_elem "thm"
        (Print.pretty_json_attribs false "{" "}" (~2) (pretty_fact ctxt fact))) named_thms
  in Print.pretty_json_elem "thms" (Print.pretty_json_attribs debug "[" "]" lvl prts) end;

(* pretty_list: (Data.T -> 'a) -> ('b -> 'a -> Pretty.T list) 
  -> bool -> int -> 'b -> string -> Data.T -> Pretty.T *)
fun pretty_list retriever prettyfier debug lvl ctxt name (data: T) =
  let
    fun wrap elem = Print.pretty_json_attribs false "{" "}" 0 [elem]
    val jattribs = Print.pretty_json_attribs debug "[" "]" lvl (map wrap (prettyfier ctxt (retriever data)))
  in Print.pretty_json_elem name jattribs end;

in 

(* bool -> int -> _ -> Data.T -> Pretty.T *)
fun pretty_json_action data 
  = Print.pretty_json_elem "action" (Pretty.quote (Pretty.str (action_of data)));

fun pretty_json_goal ctxt data 
  = Print.pretty_json_elem "term" (Pretty.quote (Syntax.pretty_term ctxt (goal_of data)));

fun pretty_json_hyps debug lvl ctxt 
  = pretty_list hyps_of pretty_hyps debug lvl ctxt "hyps";

fun pretty_json_consts debug lvl ctxt 
  = pretty_list consts_of pretty_consts debug lvl ctxt "constants";

fun pretty_json_vars debug lvl ctxt 
  = pretty_list vars_of pretty_vars debug lvl ctxt "variables";

fun pretty_json_types debug lvl ctxt 
  = pretty_list types_of pretty_types debug lvl ctxt "type variables";

fun pretty_json_methods debug lvl 
  = pretty_list methods_of pretty_methods debug lvl () "methods";

fun pretty_json_facts debug lvl data 
  = pretty_facts debug lvl (facts_of data);

(* pretty_json: bool -> data -> Pretty.T *)
fun pretty_json debug data = 
  let
    val state = state_of data;
    val ctxt = Proof.context_of (Toplevel.proof_of state)
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val goal = pretty_json_goal ctxt data;
    val hs = pretty_json_hyps debug 4 ctxt data;
    val cs = pretty_json_consts debug 4 ctxt data;
    val vs = pretty_json_vars debug 4 ctxt data;
    val Ts = pretty_json_types debug 4 ctxt data;
    val ms = pretty_json_methods debug 4 data;
    val lems = pretty_json_facts debug 4 data;
    val act = pretty_json_action data;
    val pretty_state = Print.pretty_json_attribs debug "{" "}" 2 
      [goal, hs, vs, cs, Ts, ms, lems, act]
  in Print.pretty_json_attribs debug "{" "}" 0 [Print.pretty_json_elem "state" pretty_state] end;

(* pretty_apply: string -> Toplevel.state -> Pretty.T *)
fun pretty_apply txt s = pretty_json true (apply txt s)

(* print_out: Toplevel.state -> unit *)
fun print_out s = Print.pretty_out (pretty_json true (from s));

(* string -> int -> Data.T list -> unit *)
(* TODO: Make IO operations safer to use. *)
fun write_proof dir i proof =
  let
    val name = "proof" ^ Value.print_int i ^ ".json";
    val fname = (dir ^ name);
    val _ = Ops.create_file false dir name "";
    val _ = map (Print.write_to (Print.FILE fname) o Print.string_of_pretty o pretty_json true) proof;
  in () end;

(* string -> Data.T list list -> unit *)
fun write_proofs dir proofs =
  let
    fun writting _ [] = []
      | writting n (p :: ps) = write_proof dir (n+1) p :: writting (n+1) ps
    val _ = writting 0 proofs
  in () end;

end

end;
