(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Data structure for representing Isabelle's internal data.
*)

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
      fun make_error_str ((num, err), str_opt) =
        let
          val s1 = Value.print_int num ^ ": "
          val s2 = (Print.yxml_to_string err)
        in (case str_opt of
          SOME txt => s1^s2^ ", opt: " ^ (Print.yxml_to_string txt)
          | NONE => s1^s2) end;
      val (s0, a0) = get_st_act (Sections.get_first proof);
      val (sN, aN) = get_st_act (Sections.get_last proof);
      val (s1', err') = Actions.apply a0 s0;
      val error_str = (case err' of 
        [] => NONE
        | errs => SOME (String.concatWith " | " (map make_error_str errs)))
      val _ = (case error_str of 
        SOME str => 
          raise Fail ("Data.get_env: Failed after " ^ Actions.text_of a0 ^ ".\n" ^ str)
        | NONE => ()
      );
      val sN' = Actions.apply' aN sN;
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
      |> Actions.make_filtered thy0
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
    val s' = Actions.apply' a s;
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

end;
