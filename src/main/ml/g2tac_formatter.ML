(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Pretty printer to represent Isabelle proofs as Graph2Tac training data.
*)

signature G2TAC_FORMATTER =
sig
  (* Pretty printing *)
  val pretty_state: Data.env -> Data.T -> Pretty.T
  val pretty_action: Data.T -> Pretty.T
  val pretty_ctxt: Data.env -> Data.T -> Pretty.T
  val pretty_user_state: Data.T -> Pretty.T
  val pretty_methods: Data.env -> Pretty.T
  val pretty_env: Data.env -> Pretty.T
  val pretty_step: (Data.env -> Data.T -> Pretty.T) list -> Data.env -> Data.T -> Pretty.T

  (* Formatting *)
  type format
  val all_formats: format list
  val formats_to_use: format list
  val pretty_step_of: format -> Data.T -> Data.env -> Data.T -> Pretty.T
  val pretty_step_of': format -> Data.T -> Data.env -> Data.T -> Pretty.T
  val str_number_of: format -> string
end;


structure G2tac_Formatter: G2TAC_FORMATTER =
struct

(** PRETTY PRINTING **)

(* pretty_state: Data.env -> Data.T -> Pretty.T *)
fun pretty_state env data =
  let
    val state = Data.state_of data;
    val ctxt0' = if Toplevel.is_proof state 
      then Proof.context_of (Toplevel.proof_of state) 
      else Data.ctxt_of env
    val ctxt = 
      ctxt0'
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val goal = Json_Maker.pretty_goal ctxt data;
    val hs = Json_Maker.pretty_hyps false 0 ctxt data;
    val seen = Json_Maker.pretty_proven false 0 ctxt data;
    val cs = Json_Maker.pretty_consts false 0 ctxt data;
    val vs = Json_Maker.pretty_vars false 0 ctxt data;
    val Ts = Json_Maker.pretty_types false 0 ctxt data;
    val usr_out = Json_Maker.pretty_user_state state;
    val pretty_state = Print.pretty_json_attribs false "{" "}" 0
      [usr_out, goal, hs, seen, vs, cs, Ts]
  in Print.pretty_json_attrib false 0 "step" pretty_state end;

(* pretty_state: Data.env -> Data.T -> Pretty.T *)
fun pretty_ctxt env data =
  let
    val state = Data.state_of data;
    val ctxt0' = if Toplevel.is_proof state 
      then Proof.context_of (Toplevel.proof_of state) 
      else Data.ctxt_of env
    val ctxt = 
      ctxt0'
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val goal = Json_Maker.pretty_goal ctxt data;
    val hs = Json_Maker.pretty_hyps false 0 ctxt data;
    val seen = Json_Maker.pretty_proven false 0 ctxt data;
    val cs = Json_Maker.pretty_consts false 0 ctxt data;
    val vs = Json_Maker.pretty_vars false 0 ctxt data;
    val Ts = Json_Maker.pretty_types false 0 ctxt data;
    val pretty_state = Print.pretty_json_attribs false "{" "}" 0
      [goal, hs, seen, vs, cs, Ts]
  in Print.pretty_json_attrib false 0 "context" pretty_state end;

(* pretty_action: Data.T -> Pretty.T *)
fun pretty_action data = Pretty.enclose "{" "}" 
  [Json_Maker.pretty_action data];

(* pretty_bare_action: Data.T -> Pretty.T *)
fun pretty_bare_action data = 
  let
    val backslash = "\\";
    val new_line = "\n";
    val quote = "\"";
    val to_write = 
      Data.action_of data
      |> Actions.text_of
      |> Ops.replace_all backslash (backslash ^ backslash) (* for LaTeX comments *)
      |> Ops.replace_all new_line "\\n" (* need to write explicitly "\\n" *)
      |> Ops.replace_all quote (backslash ^ quote)
  in Pretty.str to_write end;

(* pretty_user_state: Data.T -> Pretty.T *)
fun pretty_user_state data = Pretty.enclose "{" "}" 
  [Json_Maker.pretty_user_state (Data.state_of data)]

(* pretty_user_state: Data.T -> Pretty.T *)
fun pretty_bare_user_state data =
  Data.state_of data
  |> Toplevel.pretty_state
  |> Pretty.separate " " 
  |> Pretty.block
  |> Pretty.unbreakable

(* pretty_methods: Data.env -> Pretty.T *)
fun pretty_methods env = Pretty.enclose "{" "}" 
  [Json_Maker.pretty_methods false 0 env]

(* pretty_env: Data.env -> Pretty.T *)
fun pretty_env env =
  let
    val ctxt = Data.ctxt_of env
    val methods = Json_Maker.pretty_methods false 0 env;
    val ds = Json_Maker.pretty_deps false 0 ctxt env;
    val applys = Json_Maker.pretty_apply_kwrds false 0 env;
    val isars = Json_Maker.pretty_isar_kwrds false 0 env;
    val prty_prf = Print.pretty_json_attribs false "{" "}" 0
      [applys, isars, methods, ds];
  in Print.pretty_json_attrib false 0 "environment" prty_prf end;


(* (Data.env -> Data.T -> Pretty.T) list -> Data.env -> Data.T -> Pretty.T *)
fun pretty_step prettifiers env data =
  let
    val prettys =  (case length prettifiers of 
      2 => 
        [(Ops.ith 0 prettifiers) env data, 
        Pretty.str "OUTPUT", 
        (Ops.ith 1 prettifiers) env data,
        Pretty.str "<END>", 
        Pretty.fbrk]
      | 3 => 
        [(Ops.ith 0 prettifiers) env data, 
        Pretty.str "SEP", 
        (Ops.ith 1 prettifiers) env data,
        Pretty.str "OUTPUT", 
        (Ops.ith 2 prettifiers) env data,
        Pretty.str "<END>", 
        Pretty.fbrk]
      | _ =>
        [pretty_state env data, 
        Pretty.str "OUTPUT", 
        pretty_action data, 
        Pretty.str "<END>", 
        Pretty.fbrk])
  in Ops.intersp [Pretty.brk 1] prettys |> Pretty.block end;


(** FORMATTING **)

val pretty1 = [
  (fn _ => pretty_bare_user_state), 
  (fn _ => pretty_bare_action)];

val pretty1' = [
  pretty_state, 
  (fn _ => pretty_action)];

fun pretty2 prev_data =
  [(fn _ => fn _ => pretty_user_state prev_data),
  (fn _ => pretty_user_state),
  (fn _ => pretty_action)];

fun pretty2' prev_data =
  [(fn env => fn _ => pretty_state env prev_data),
  pretty_state,
  (fn _ => pretty_action)];

val pretty3 = 
  [pretty_ctxt,
  (fn _ => pretty_user_state),
  (fn _ => pretty_action)];

val pretty4 =
  [(fn env => fn _ => pretty_methods env),
  (fn _ => pretty_user_state),
  (fn _ => pretty_action)];

val pretty4' =
  [(fn env => fn _ => pretty_methods env),
  pretty_state,
  (fn _ => pretty_action)];

val pretty5 =
  [(fn env => fn _ => pretty_env env),
  (fn _ => pretty_user_state),
  (fn _ => pretty_action)]

val pretty5' =
  [(fn env => fn _ => pretty_env env),
  pretty_state,
  (fn _ => pretty_action)]


datatype format = 
  G2TAC1    (* state OUTPUT action <END> \n  *)
  | G2TAC2  (* {state_t-1} SEP {state_t} OUTPUT {action} <END> \n   *)
  | G2TAC3  (* {context, i.e. constants, variables, ...} SEP {state} OUTPUT {action} <END> \n   *)
  | G2TAC4  (* {list of available tactics} SEP {state} OUTPUT {action} <END> \n   *)
  | G2TAC5  (* {environment} SEP {state_t} OUTPUT {action} <END> \n   *)

val formats_to_use = [G2TAC1]
val all_formats = [G2TAC1, G2TAC2, G2TAC3, G2TAC4, G2TAC5]

(* pretty_step_of: format -> Data.T -> Data.env -> Data.T -> Pretty.T *)
fun pretty_step_of format prev_data = (case format of 
  G2TAC1 => pretty_step pretty1
  | G2TAC2 => pretty_step (pretty2 prev_data)
  | G2TAC3 => pretty_step pretty3
  | G2TAC4 => pretty_step pretty4
  | G2TAC5 => pretty_step pretty5)

(* pretty_step_of': format -> Data.T -> Data.env -> Data.T -> Pretty.T *)
fun pretty_step_of' format prev_data = (case format of 
  G2TAC1 => pretty_step pretty1'
  | G2TAC2 => pretty_step (pretty2' prev_data)
  | G2TAC3 => pretty_step pretty3
  | G2TAC4 => pretty_step pretty4'
  | G2TAC5 => pretty_step pretty5')

(* str_number_of: g2tac_formats -> string *)
fun str_number_of format = (case format of 
  G2TAC1 => Value.print_int 1
  | G2TAC2 => Value.print_int 2
  | G2TAC3 => Value.print_int 3
  | G2TAC4 => Value.print_int 4
  | G2TAC5 => Value.print_int 5)

end;
