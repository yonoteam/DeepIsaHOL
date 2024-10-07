
signature DATA =
sig
  (* Environment *)
  type env
  val s1_of: env -> Toplevel.state
  val ctxt_of: env -> Proof.context
  val mthds_of: env -> string list
  val deps_of: env -> (Thm_Name.T * thm) list
  val grp_cmmnds_of: env -> (string * string list) list
  val get_env: ('a -> Toplevel.state * Actions.T) -> 'a Sections.T -> env

  (* Data.T *)
  type T
  val state_of: T -> Toplevel.state
  val action_of: T -> Actions.T
  val goal_of: T -> term
  val hyps_of: T -> term list
  val consts_of: T -> term list
  val vars_of: T -> term list
  val types_of: T -> typ list
  val proven_of: T -> term list

  (* Extraction *)
  val init: env -> Toplevel.state * Actions.T -> T
  val next: env -> (Toplevel.state * Actions.T) -> T -> T
  val extract_proof: (Toplevel.state * Actions.T) Sections.T -> env * T Sections.T
  val extract_proofs: theory -> string -> (env * (T Sections.T)) list
  val from: Toplevel.state -> T
  val apply: string -> T -> T

  (* Pretty printing *)
  val pretty_json_action: T -> Pretty.T
  val pretty_json_goal: Proof.context -> T -> Pretty.T
  val pretty_json_hyps: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_proven: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_consts: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_vars: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_types: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_deps: bool -> int -> Proof.context -> env -> Pretty.T
  val pretty_json_methods: bool -> int -> env -> Pretty.T
  val pretty_json_apply_kwrds: bool -> int -> env -> Pretty.T
  val pretty_json_isar_kwrds: bool -> int -> env -> Pretty.T
  val pretty_json_step: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_proof: bool -> int -> env -> T Sections.T -> Pretty.T

  (* interface *)
  val extract: theory * string -> string
  val write_proof: string -> (env * T Sections.T) -> unit
  val write_proofs: (string * theory * string) -> unit
end;


structure Data: DATA =
struct

(** ENVIRONMENT **)

datatype env = Env of 
  {s1: Toplevel.state,
  context: Proof.context, 
  methods: string list, 
  deps: (Thm_Name.T * thm) list, 
  grp_cmmnds: (string * string list) list}

fun rep_env (Env e) = e;
fun s1_of e = #s1 (rep_env e);
fun ctxt_of e = #context (rep_env e);
fun mthds_of e = #methods (rep_env e);
fun deps_of e = #deps (rep_env e);
fun grp_cmmnds_of e = #grp_cmmnds (rep_env e);

fun get_env get_st_act proof =
  if Sections.get_top_tag proof = SOME Sections.proof_tag then
    let 
      val (s0, a0) = get_st_act (Sections.get_first proof);
      val (sN, aN) = get_st_act (Sections.get_last proof);
      val (s1', err') = Actions.apply_safe a0 s0;
      val _ = (case err' of 
        SOME (_, str) => raise Fail ("Data.get_env: Failed after " ^ Actions.text_of a0 ^ ".\n" ^ str)
        | NONE => ()
      );
      val sN' = Actions.apply aN sN;
      val ctxtN' = Toplevel.context_of sN';
      val thyN' = Toplevel.theory_of sN';
      val diffs = Get.new_thms s0 sN';
    in Env {s1 = s1',
      context = ctxtN', 
      methods = Get.methods (Toplevel.context_of s0), 
      deps = Get.deps thyN' (maps snd diffs), 
      grp_cmmnds = Get.grouped_commands thyN'} 
    end
  else raise Fail "Data.get_env: Input not a proof block.";


(** DATA **)

datatype T = Data of {
  state: Toplevel.state, 
  action: Actions.T, 
  goal: term,
  hyps: term list,
  consts: term list, 
  vars: term list,
  types: typ list, 
  proven: term list};

(* Data.T -> _ *)
fun rep_data (Data d) = d;
fun state_of d = #state (rep_data d);
fun action_of d = #action (rep_data d);
fun goal_of d = #goal (rep_data d);
fun hyps_of d = #hyps (rep_data d);
fun consts_of d = #consts (rep_data d);
fun vars_of d = #vars (rep_data d);
fun types_of d = #types (rep_data d);
fun proven_of d = #proven (rep_data d);


(** FILE EXTRACTION **)

(* TODO: Generalise? *)
(* init: Data.env -> Toplevel.state * Actions.T -> Data.T *)
fun init env (s, a) = 
  let
    val s' = s1_of env;
    val _ = if Toplevel.is_proof s' then () 
      else raise Fail "Data.init: Non-pre-proof input";
    val prop = Proof.goal (Toplevel.proof_of s');
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val xs = Get.vars goal_hyps;
    val cs = Get.consts goal_hyps;
    val Ts = Get.atomic_types goal_hyps;
  in Data {
    state = s,
    action = a,
    goal = hd goal_hyps,
    hyps = tl goal_hyps,
    consts = cs,
    vars = xs,
    types = Ts,
    proven = []}
  end;

(* next: Data.env -> (Toplevel.state * Actions.T) -> Data.T *)
fun next env (s_new, act_to_do) old_data = 
  let val cmmnds = grp_cmmnds_of env in
  if Actions.on_kind (Seps.is_apply_step cmmnds) act_to_do then
    (let
      val prop = Proof.goal (Toplevel.proof_of s_new);
      val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
      val new_goal = hd goal_hyps;
      val new_proven = 
        if Actions.on_kind (Seps.is_apply_step cmmnds) (action_of old_data)
        then (goal_of old_data) :: proven_of old_data
        else proven_of old_data;
      val xs = Get.vars goal_hyps;
      val cs = Get.consts goal_hyps;
      val Ts = Get.atomic_types goal_hyps;
      val hs = tl goal_hyps
    in let
      val result = Data {
      state = s_new, 
      action = act_to_do,
      goal = new_goal,
      hyps = hs,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven}
     in result end
    end)
  else 
    (let
      val prop = Proof.goal (Toplevel.proof_of s_new);
      val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
      val new_goal = hd goal_hyps;
      val facts_to_add = 
        if Actions.on_kind (Seps.ends_proof cmmnds) (action_of old_data) 
        then Get.new_thms (s1_of env) s_new
        else []
      val new_proven = (map Thm.prop_of (maps snd facts_to_add)) @ (proven_of old_data)
      val xs = Get.vars goal_hyps;
      val cs = Get.consts goal_hyps;
      val Ts = Get.atomic_types goal_hyps;
    in Data {
      state = s_new, 
      action = act_to_do,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven}
    end) end;

(* extract_proof: (Toplevel.state * Actions.T) Sections.T -> Data.T Sections.T *)
fun extract_proof proof =
  let
    val (s0, a0) = Sections.get_first proof;
    val env = get_env I proof;
    val data0 = init env (s0, a0);
  in (env, Sections.do_one_then_fold true data0 (next env) proof) end;

(* extract_proofs: theory -> string -> (Data.env * (Data.T Sections.T)) list *)
fun extract_proofs thy0 thy_text =
  let
    val state0 = Toplevel.make_state NONE;
    val proofs =
      thy_text
      |> Actions.make' thy0
      |> Actions.segmentate state0 (Actions.theory_separators thy0)
      |> Sections.get_topmost Sections.proof_tag
      |> map extract_proof;
  in proofs end;


(** INTERACTIVE EXTRACTION **)

(* from: Toplevel.state -> Data.T *)
fun from s =
  let
    val prop = Proof.goal (Toplevel.proof_of s);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val obj = hd goal_hyps;
    val xs = Get.vars goal_hyps;
    val cs = Get.consts goal_hyps;
    val Ts = Get.atomic_types goal_hyps;
  in Data {
    state = s, 
    action = Actions.void,
    goal = obj,
    hyps = tl goal_hyps,
    consts = cs,
    vars = xs,
    types = Ts,
    proven = []}
  end;

(* apply: string -> Data.T -> Data.T *)
fun apply txt old_data = 
  let 
    val s = state_of old_data;
    val a = Actions.make_one (Toplevel.theory_of s) txt;
    val s' = Actions.apply a s;
    val cmmnds = Get.grouped_commands (Toplevel.theory_of s');
    val prop = Proof.goal (Toplevel.proof_of s');
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val new_goal = hd goal_hyps;
    val xs = Get.vars goal_hyps;
    val cs = Get.consts goal_hyps;
    val Ts = Get.atomic_types goal_hyps;
  in
  if Actions.on_kind ((Seps.is_apply_step cmmnds) andf Pred.neg (Seps.ends_proof cmmnds)) a 
  then
    let      
      val new_proven = (goal_of old_data) :: proven_of old_data;
    in Data {
      state = s', 
      action = a,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven}
    end
  else 
    let
      val facts_to_add = 
        if Actions.on_kind (Seps.ends_proof cmmnds) a 
        then Get.new_thms s s'
        else []
      val new_proven = (map Thm.prop_of (maps snd facts_to_add)) @ (proven_of old_data)
    in Data {
      state = s', 
      action = a,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven}
    end end;


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

(* pretty_keywords: unit -> string list -> Pretty.T list *)
val pretty_keywords = (map (Print.pretty_json_elem "name" o Pretty.quote o Pretty.str));

(* pretty_hyps: Proof.context -> string list -> Pretty.T list *)
fun pretty_hyps ctxt 
  = (map (Print.pretty_json_elem "term" o Pretty.quote o Pretty.unbreakable o Syntax.pretty_term ctxt));

(* pretty_fact: Proof.context -> (string * int) * thm -> Pretty.T list *)
fun pretty_fact ctxt ((name, i),thm) = [
  Print.pretty_json_elem "name" (
    Pretty.quote (Pretty.block [
      Pretty.marks_str (#1 (Proof_Context.markup_extern_fact ctxt name), name),
      Pretty.str (Thm_Name.print (name, i))])
  ),
  Print.pretty_json_elem "term" (Pretty.quote (Thm.pretty_thm ctxt thm))
  ];

(* pretty_facts: Proof.context -> bool -> int -> ((string * int) * thm) list -> Pretty.T *)
fun pretty_facts ctxt debug lvl categ_name named_thms =
  let
    fun jsonify fact = Pretty.enclose "{" "}"
      [Print.pretty_json_elem "thm"
        (Print.pretty_json_attribs false "{" "}" (~2) (pretty_fact ctxt fact))]
    val prts = map jsonify named_thms
  in Print.pretty_json_elem categ_name (Print.pretty_json_attribs debug "[" "]" lvl prts) end;

(* pretty_list: ('record -> 'a) -> ('a -> Pretty.T list) 
  -> bool -> int -> string -> 'record -> Pretty.T *)
fun pretty_list retriever prettyfier debug lvl name record =
  let
    fun wrap elem = Print.pretty_json_attribs false "{" "}" 0 [elem]
    val jattribs = Print.pretty_json_attribs debug "[" "]" lvl (map wrap (prettyfier (retriever record)))
  in Print.pretty_json_elem name jattribs end;

in 

(* bool -> int -> _ -> Data.T -> Pretty.T *)
fun pretty_json_action data = 
  let
    val backslash = "\\";
    val quote_mark = "\"";
    val new_line = "\n";
    val to_write = 
      action_of data
      |> Actions.text_of
      |> Ops.replace_all quote_mark (backslash ^ quote_mark)
      |> Ops.replace_all new_line "\\n"
  in Print.pretty_json_elem "action" (Pretty.quote (Pretty.str to_write)) end;

fun pretty_user_output state =
  let
    val output_str = Get.user_output state
    val backslash = "\\";
    val quote_mark = "\"";
    val new_line = "\n";
    val to_write = 
      Ops.replace_all quote_mark (backslash ^ quote_mark) output_str
      |> Ops.replace_all new_line "\\n"
  in Print.pretty_json_elem "user_output" (Pretty.quote (Pretty.str to_write)) end;

fun pretty_json_goal ctxt data = 
  Print.pretty_json_elem "term" (Pretty.quote (Pretty.unbreakable (Syntax.pretty_term ctxt (goal_of data))));

fun pretty_json_hyps debug lvl ctxt = 
  pretty_list hyps_of (pretty_hyps ctxt) debug lvl "hyps";

fun pretty_json_proven debug lvl ctxt = 
  pretty_list proven_of (pretty_hyps ctxt) debug lvl "proven";

fun pretty_json_consts debug lvl ctxt = 
  pretty_list consts_of (pretty_consts ctxt) debug lvl "constants";

fun pretty_json_vars debug lvl ctxt = 
  pretty_list vars_of (pretty_vars ctxt) debug lvl "variables";

fun pretty_json_types debug lvl ctxt = 
  pretty_list types_of (pretty_types ctxt) debug lvl "type variables";

fun pretty_json_apply_kwrds debug lvl =
  let
    fun retriever env =
      let
        val cmnds = grp_cmmnds_of env
        val is_apply = Seps.is_apply_step cmnds orf Pred.is "proof"
      in filter is_apply (maps snd cmnds) end;
  in pretty_list retriever pretty_keywords debug lvl "apply_kwrds" end;

fun pretty_json_isar_kwrds debug lvl =
  let
    fun retriever env =
      let
        val cmnds = grp_cmmnds_of env
        val is_apply = Seps.is_isar_step cmnds
      in filter is_apply (maps snd cmnds) end;
  in pretty_list retriever pretty_keywords debug lvl "isar_kwrds" end; 

fun pretty_json_methods debug lvl = 
  pretty_list mthds_of pretty_keywords debug lvl "methods";

fun pretty_json_deps debug lvl ctxt env =
  pretty_facts ctxt debug lvl "deps" (deps_of env)

(* pretty_json_step: bool -> int -> data -> Pretty.T *)
fun pretty_json_step debug lvl ctxt0 data = 
  let
    val state = state_of data;
    val ctxt0' = if Toplevel.is_proof state 
      then Proof.context_of (Toplevel.proof_of state) 
      else ctxt0
    val ctxt = 
      ctxt0'
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val goal = pretty_json_goal ctxt data;
    val hs = pretty_json_hyps debug (lvl+4) ctxt data;
    val seen = pretty_json_proven debug (lvl+4) ctxt data;
    val cs = pretty_json_consts debug (lvl+4) ctxt data;
    val vs = pretty_json_vars debug (lvl+4) ctxt data;
    val Ts = pretty_json_types debug (lvl+4) ctxt data;
    val usr_out = pretty_user_output state;
    val act = pretty_json_action data;
    val pretty_state = Print.pretty_json_attribs debug "{" "}" (lvl+2) 
      [act, usr_out, goal, hs, seen, vs, cs, Ts]
  in Print.pretty_json_attrib debug lvl "step" pretty_state end;

(* pretty_json_proof: bool -> int -> Data.env -> Data.T Sections.T -> Pretty.T *)
fun pretty_json_proof debug lvl env proof =
  let
    val ctxt = ctxt_of env
    val trace = Sections.recover_trace proof;
    val prt_steps =
      map (pretty_json_step debug (lvl+4) ctxt) trace
      |> Print.pretty_json_attribs debug "[" "]" (lvl+4)
      |> Print.pretty_json_elem "steps"
    val methods = pretty_json_methods debug (lvl+4) env;
    val ds = pretty_json_deps debug (lvl+4) ctxt env;
    val applys = pretty_json_apply_kwrds debug (lvl+4) env;
    val isars = pretty_json_isar_kwrds debug (lvl+4) env;
    val prty_prf = Print.pretty_json_attribs debug "{" "}" (lvl+2) 
      [prt_steps, applys, isars, methods, ds];
  in Print.pretty_json_attrib debug lvl "proof" prty_prf end;

(* json_proof: (Data.env * Data.T Sections.T) -> string *)
fun json_proof (env, proof) =
  let
    val backslash = "\\";
    val start_quote = backslash ^ "<";
    fun rewrite_quote txt = 
      Ops.replace_all start_quote (backslash ^ start_quote) txt
    val prf = 
      pretty_json_proof true 0 env proof
      |> Print.string_of_pretty
      |> rewrite_quote
  in prf end;

end;


(** INTERFACT **)

(* val _ = ML_system_pp (fn _ => fn _ 
  => Pretty.to_polyml o pretty_json true 0 (Context.the_local_context ())) *)

(* extract: theory * string -> string *)
fun extract (thy0, thy_text) =
  extract_proofs thy0 thy_text
  |> map json_proof
  |> String.concatWith " ISA_RL_SEP "

(* write_proof: string -> (Data.env * Data.T Sections.T) -> unit *)
fun write_proof file_path proof =
  json_proof proof
  |> Print.write_to (Print.FILE file_path)

(* write_proofs: (string * theory * string) -> unit *)
fun write_proofs (write_dir, thy0, thy_text) =
  let 
    val proofs = extract_proofs thy0 thy_text;
    fun fname j = ("proof" ^ Value.print_int j ^ ".json");
    fun write_next proof i =
      let
        val file_path = OS.Path.joinDirFile {dir=write_dir, file=fname i}
        val _ = write_proof file_path proof
      in i+1 end;
    val _ = fold write_next proofs 0
  in () end;

end;
