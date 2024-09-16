
signature DATA =
sig
  (* Environment *)
  type env
  val s1_of: env -> Toplevel.state
  val ctxt_of: env -> Proof.context
  val mthds_of: env -> string list
  val deps_of: env -> (Thm_Name.T * thm) list
  val grp_cmmnds_of: env -> (string * string list) list
  val get_env: (Toplevel.state * Actions.T) Sections.T -> env

  (* Data.T *)
  type T
  val state_of: T -> Toplevel.state
  val action_of: T -> Actions.T
  val goal_of: T -> term
  val hyps_of: T -> term list
  val consts_of: T -> term list
  val vars_of: T -> term list
  val types_of: T -> typ list
  val keywords_of: T -> string list
  val methods_of: T -> string list
  val facts_of: T -> (Thm_Name.T * thm) list

  (* Extraction *)
  val init: env -> Toplevel.state * Actions.T -> T
  val next: env -> (Toplevel.state * Actions.T) -> T -> T
  val extract_proof: (Toplevel.state * Actions.T) Sections.T -> env * T Sections.T
  val extract_proofs: theory -> string -> (env * (T Sections.T)) list
  val from: Toplevel.state -> T
  val apply: string -> T -> T
  val extract: theory * string -> string

  (* Pretty printing *)
  val pretty_json_action: T -> Pretty.T
  val pretty_json_goal: Proof.context -> T -> Pretty.T
  val pretty_json_hyps: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_consts: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_vars: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_types: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_facts: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_methods: bool -> int -> T -> Pretty.T
  val pretty_json: bool -> int -> Proof.context -> T -> Pretty.T
  val pretty_json_proof: bool -> int -> env -> T Sections.T -> Pretty.T

  (* Writing *)
  val write_proofs: string -> ((Toplevel.state * Actions.T) Sections.T) list -> unit
  val retrieve_from_to: string -> string -> unit
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

fun get_env proof =
  if Sections.get_top_tag proof = SOME Sections.proof_tag then
    let 
      val (s0, a0) = Sections.get_first proof;
      val (sN, aN) = Sections.get_last proof;
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
  proven: term list,
  keywords: string list, 
  methods: string list,
  facts: (Thm_Name.T * thm) list};

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
fun keywords_of d = #keywords (rep_data d);
fun methods_of d = #methods (rep_data d);
fun facts_of d = #facts (rep_data d);


(** FILE EXTRACTION **)

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
    val cmmnds = grp_cmmnds_of env;
    val kwrds = filter (Seps.is_apply_step cmmnds orf Pred.is "proof") (maps snd cmmnds);
  in Data {
    state = s, 
    action = a,
    goal = hd goal_hyps,
    hyps = tl goal_hyps,
    consts = cs,
    vars = xs,
    types = Ts,
    proven = [],
    keywords = kwrds,
    methods = mthds_of env,
    facts = []}
  end;

(* next: Data.env -> (Toplevel.state * Actions.T) -> Data.T *)
fun next env (s_new, act_to_do) old_data = 
  let val cmmnds = grp_cmmnds_of env; in
  if Actions.on_kind (Seps.is_apply_step cmmnds) act_to_do then
    let
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
      val kwrds = filter (Seps.is_apply_step cmmnds orf Pred.is "proof") (maps snd cmmnds);
    in Data {
      state = s_new, 
      action = act_to_do,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven,
      keywords = kwrds,
      methods = mthds_of env,
      facts = facts_of old_data}
    end
  else 
    let
      val prop = Proof.goal (Toplevel.proof_of s_new);
      val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
      val new_goal = hd goal_hyps;
      val facts_to_add = 
        if Actions.on_kind (Seps.ends_proof cmmnds) (action_of old_data) 
        then Get.new_thms (s1_of env) s_new
        else []
      val new_proven = (map Thm.prop_of (maps snd facts_to_add)) @ (proven_of old_data)
      val new_facts = Get.named_thms (facts_to_add) @ (facts_of old_data)
      val xs = Get.vars goal_hyps;
      val cs = Get.consts goal_hyps;
      val Ts = Get.atomic_types goal_hyps;
      val kwrds = filter (Seps.is_isar_step cmmnds) (maps snd cmmnds);
    in Data {
      state = s_new, 
      action = act_to_do,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven,
      keywords = kwrds,
      methods = [],
      facts = new_facts}
    end end;

(* extract_proof: (Toplevel.state * Actions.T) Sections.T -> Data.T Sections.T *)
fun extract_proof proof =
  let
    val (s0, a0) = Sections.get_first proof;
    val env = get_env proof;
    val data0 = init env (s0, a0);
  in (env, Sections.do_one_then_fold true data0 (next env) proof) end;

(* extract_proofs: theory -> string -> (Data.env * (Data.T Sections.T)) list *)
fun extract_proofs thy0 thy_file =
  let
    val state0 = Toplevel.make_state (SOME thy0);
    val proofs =
      thy_file
      |> Ops.read_file
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
    val thy = Toplevel.theory_of s;
    val ctxt = Toplevel.context_of s;
    val cmmnds = maps snd (Get.grouped_commands thy);
    val ms = Get.methods ctxt
    val sought = [] (* Temp.query_thms ctxt 20 obj *);
  in Data {
    state = s, 
    action = Actions.void,
    goal = obj,
    hyps = tl goal_hyps,
    consts = cs,
    vars = xs,
    types = Ts,
    proven = [],
    keywords = cmmnds,
    methods = ms,
    facts = sought}
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
      val kwrds = filter (Seps.is_apply_step cmmnds orf Pred.is "proof") (maps snd cmmnds);
      val ctxt = Toplevel.context_of s';
      val ms = Get.methods ctxt
    in Data {
      state = s', 
      action = a,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven,
      keywords = kwrds,
      methods = ms,
      facts = facts_of old_data}
    end
  else 
    let
      val facts_to_add = 
        if Actions.on_kind (Seps.ends_proof cmmnds) a 
        then Get.new_thms s s'
        else []
      val new_proven = (map Thm.prop_of (maps snd facts_to_add)) @ (proven_of old_data)
      val new_facts = Get.named_thms (facts_to_add) @ (facts_of old_data);
      val kwrds = filter (Seps.is_isar_step cmmnds) (maps snd cmmnds);
    in Data {
      state = s', 
      action = a,
      goal = new_goal,
      hyps = tl goal_hyps,
      consts = cs,
      vars = xs,
      types = Ts,
      proven = new_proven,
      keywords = kwrds,
      methods = [],
      facts = new_facts}
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
  = (map (Print.pretty_json_elem "term" o Pretty.quote o Syntax.pretty_term ctxt));

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
fun pretty_facts ctxt debug lvl named_thms =
  let
    fun jsonify fact = Pretty.enclose "{" "}"
      [Print.pretty_json_elem "thm"
        (Print.pretty_json_attribs false "{" "}" (~2) (pretty_fact ctxt fact))]
    val prts = map jsonify named_thms
  in Print.pretty_json_elem "facts" (Print.pretty_json_attribs debug "[" "]" lvl prts) end;

(* pretty_list: (Data.T -> 'a) -> ('a -> Pretty.T list) 
  -> bool -> int -> string -> Data.T -> Pretty.T *)
fun pretty_list retriever prettyfier debug lvl name (data: T) =
  let
    fun wrap elem = Print.pretty_json_attribs false "{" "}" 0 [elem]
    val jattribs = Print.pretty_json_attribs debug "[" "]" lvl (map wrap (prettyfier (retriever data)))
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
      |> Ops.replace_all new_line " "
  in Print.pretty_json_elem "action" (Pretty.quote (Pretty.str to_write)) end;

fun pretty_json_goal ctxt data = 
  Print.pretty_json_elem "term" (Pretty.quote (Syntax.pretty_term ctxt (goal_of data)));

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

fun pretty_json_keywords debug lvl = 
  pretty_list keywords_of pretty_keywords debug lvl "keywords";

fun pretty_json_methods debug lvl = 
  pretty_list methods_of pretty_keywords debug lvl "methods";

fun pretty_json_facts debug lvl ctxt data = 
  pretty_facts ctxt debug lvl (facts_of data);

(* pretty_json: bool -> int -> data -> Pretty.T *)
fun pretty_json debug lvl ctxt0 data = 
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
    val ks = pretty_json_keywords debug (lvl+4) data;
    val ms = pretty_json_methods debug (lvl+4) data;
    val lems = pretty_json_facts debug (lvl+4) ctxt data;
    val act = pretty_json_action data;
    val pretty_state = Print.pretty_json_attribs debug "{" "}" (lvl+2) 
      [act, goal, hs, seen, vs, cs, Ts, ks, ms, lems]
  in Print.pretty_json_attrib debug lvl "step" pretty_state end;

(* pretty_json_proof: bool -> int -> Data.env -> Data.T Sections.T -> Pretty.T *)
fun pretty_json_proof debug lvl env proof =
  let
    val ctxt = ctxt_of env
    val trace = Sections.recover_trace proof;
    val prt_prf =
      map (pretty_json debug (lvl+2) ctxt) trace
      |> Print.pretty_json_attribs debug "[" "]" lvl
      |> Print.pretty_json_attrib false lvl "proof"
  in prt_prf end;

(* extract: theory * string -> string *)
fun extract (thy0, thy_file) =
  let 
    val proofs = extract_proofs thy0 thy_file;
    fun string_proof (env, proof) = 
      pretty_json_proof true 0 env proof
      |> Print.string_of_pretty
  in String.concatWith " ISA_RL_SEP " (map string_proof proofs) end;

end;


(** WRITING OPERATIONS **)

val _ = ML_system_pp (fn _ => fn _ 
  => Pretty.to_polyml o pretty_json true 0 (Context.the_local_context ()))

(* write_proof: string -> int -> (Toplevel.state * Actions.T) Sections.T -> unit *)
fun write_proof' dir i proof =
  let
    (* IO operartions *)
    val env = get_env proof;
    val thy_name = Context.theory_name {long=false} (Toplevel.theory_of (s1_of env))
    val dir_name = dir ^ "/" ^ thy_name ^ "/" ^ "proof" ^ Value.print_int i;
      (* extra "/"s are no issue for OS.FileSys.mkDir *)
    val _ = Ops.create_directory {force=true} dir_name
        handle Fail txt => Exn.reraise (Fail txt);
    fun fname j = ("step" ^ Value.print_int j ^ ".json");

    (* Data.T operartions *)
    val start = init env (Sections.get_first proof);
    val proof_data = Sections.do_one_then_fold true start (next env) proof
    val trace = Sections.recover_trace proof_data;
    
    (* String operartions *)
    val backslash = "\\";
    val start_quote = backslash ^ "<";
    fun rewrite_quote txt = Ops.replace_all start_quote (backslash ^ start_quote) txt
    
    (* Main *)
    fun write_next data j =
      let
        val _ = Ops.create_file {force=false} dir_name (fname j) "";
        val _ =
          data
          |> pretty_json true 0 (ctxt_of env)
          |> Print.string_of_pretty
          |> rewrite_quote
          |> Print.write_to (Print.FILE (OS.Path.joinDirFile {dir=dir_name, file=fname j}))
      in j+1 end;
    val _ = if null trace 
      then raise Fail "Data.write_proof: empty proof" 
      else fold write_next (tl trace) (write_next (hd trace) 0)
  in () end;

(* write_proof: string -> int -> (Toplevel.state * Actions.T) Sections.T -> unit *)
fun write_proof dir i proof =
  let
    (* IO operartions *)
    val env = get_env proof;
    val thy_name = Context.theory_name {long=false} (Toplevel.theory_of (s1_of env))
    val dir_name = dir ^ "/" ^ thy_name ^ "/";
      (* extra "/"s are no issue for OS.FileSys.mkDir *)
    val _ = Ops.create_directory {force=true} dir_name
        handle Fail txt => Exn.reraise (Fail txt);
    fun fname j = ("proof" ^ Value.print_int j ^ ".json");

    (* Data.T operartions *)
    val start = init env (Sections.get_first proof);
    val proof_data = Sections.do_one_then_fold true start (next env) proof
    val prt_prf = pretty_json_proof true 0 env proof_data;
    
    (* String operartions *)
    val backslash = "\\";
    val start_quote = backslash ^ "<";
    fun rewrite_quote txt = Ops.replace_all start_quote (backslash ^ start_quote) txt
    
    val _ =
      prt_prf
      |> Print.string_of_pretty
      |> rewrite_quote
      |> Print.write_to (Print.FILE (OS.Path.joinDirFile {dir=dir_name, file=fname i}))
  in () end;

(* write_proofs: string -> ((Toplevel.state * Actions.T) Sections.T) list -> unit *)
fun write_proofs dir proofs =
  let
    fun writting _ [] = []
      | writting n (p :: ps) = write_proof dir n p :: writting (n+1) ps
    val _ = writting 0 proofs
  in () end;

(* retrieve_from_to: string -> string -> unit *)
fun retrieve_from_to thy_file dir =
  let
    val thy0 = Get.thy_from_file thy_file;
    val state0 = Toplevel.make_state (SOME thy0);
    val proofs =
      thy_file
      |> Ops.read_file
      |> Actions.make' thy0
      |> Actions.segmentate state0 (Actions.theory_separators thy0)
      |> Sections.get_topmost Sections.proof_tag;
  in write_proofs dir proofs end;

end;
